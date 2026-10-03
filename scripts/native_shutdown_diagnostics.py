"""Opt-in, container-local native shutdown backtrace; no pipeline mutations."""
from pathlib import Path
import subprocess


# These are read-only observations while gdb has stopped every inferior thread.
# Candidate register addresses are NOT asserted to be mutexes: the disassembly
# and the named argument (if debug information exists) must prove that first.
MUTEX_DIAGNOSTIC = r'''
import gdb
threads = gdb.selected_inferior().threads()
print("F5_MUTEX_SNAPSHOT_BEGIN")
for thread in threads:
    thread.switch()
    frame = gdb.newest_frame()
    depth = 0
    while frame is not None and depth < 12:
        name = frame.name() or ""
        if "mutex_lock" in name or "lll_lock_wait" in name:
            frame.select()
            print("F5_MUTEX_WAIT thread=%s ptid=%s name=%s frame=%s pc=%s" %
                  (thread.num, thread.ptid, thread.name, name, hex(frame.pc())))
            addresses = {}
            try:
                addresses["debug_argument_mutex"] = int(frame.read_var("mutex"))
            except (gdb.error, ValueError):
                pass
            for register in ("rdi", "rbx", "r12", "r13"):
                try:
                    addresses["register_" + register] = int(frame.read_register(register))
                except (gdb.error, ValueError):
                    pass
            print("F5_REGISTER_CANDIDATES %s" % addresses)
            for source, address in addresses.items():
                if address < 4096:
                    continue
                try:
                    raw = bytes(gdb.selected_inferior().read_memory(address, 40))
                    words = [int.from_bytes(raw[i:i+4], "little", signed=True)
                             for i in range(0, 24, 4)]
                    print("F5_MUTEX_MEMORY candidate=%s address=%s words=%s UNPROVEN_UNTIL_ARGUMENT_OR_DISASSEMBLY_MATCH" %
                          (source, hex(address), words))
                except gdb.error as error:
                    print("F5_MEMORY_UNAVAILABLE %s %s" % (source, error))
            try:
                print(gdb.execute("disassemble /r " + name, to_string=True))
            except gdb.error as error:
                print("F5_DISASSEMBLY_UNAVAILABLE %s" % error)
        frame = frame.older()
        depth += 1
print("F5_MUTEX_SNAPSHOT_END")
'''


def debugger_arguments():
    """No auto-loading, memory mutation, signal injection, or argument dumps."""
    return ["--data-directory=/diag/gdb-data", "-nx", "-batch",
        "-iex", "set auto-load off", "-iex", "set debuginfod enabled off",
        "-ex", "set pagination off", "-ex", "set print frame-arguments none",
        "-ex", "attach 1", "-ex", "thread apply all bt",
        "-ex", "python exec(" + repr(MUTEX_DIAGNOSTIC) + ")", "-ex", "detach"]


def debugger_command(command, image):
    result = list(command)
    at = result.index(image)
    result[at:at] = ["--cap-add=SYS_PTRACE", "--security-opt=seccomp=unconfined",
        "-v", "/usr/bin/gdb:/diag/gdb:ro",
        "-v", "/usr/lib/x86_64-linux-gnu:/diag/lib:ro",
        "-v", "/usr/share/gdb:/diag/gdb-data:ro"]
    return result


def shutdown_backtrace(container, output: Path):
    # Read-only libraries are used only by gdb, not by the inference process.
    # PID 1 must be the already-owned native binary, never a different process.
    exe = subprocess.check_output(["docker", "exec", container, "readlink", "/proc/1/exe"], text=True).strip()
    if exe != "/workspace/proto-app":
        raise RuntimeError("refusing debugger: PID 1 is not the candidate binary")
    # Preserve process/thread, socket and broker state BEFORE attaching/killing.
    # Deliberately exclude environment/command arguments, which may contain RTSP
    # credentials, and do not capture unrestricted docker inspect output.
    state_command = ["docker", "exec", "--user", "root", container, "sh", "-c",
        "for p in /proc/1/status /proc/1/maps; do echo FILE=$p; cat \"$p\"; done; "
        "for t in /proc/1/task/*; do echo TASK=$t; "
        "for f in comm status wchan syscall stack; do echo FIELD=$f; cat \"$t/$f\" 2>&1; done; done"]
    with output.with_suffix(".proc.log").open("x") as log:
        subprocess.run(state_command, stdout=log, stderr=subprocess.STDOUT, timeout=10)
    with output.with_suffix(".container.log").open("x") as log:
        for args in (["docker", "inspect", "--format",
                      "{{.Name}} pid={{.State.Pid}} status={{.State.Status}} started={{.State.StartedAt}} exit={{.State.ExitCode}}", container],
                     ["docker", "top", container, "-eLo", "pid,tid,comm,state,wchan:32"],
                     ["ss", "-Htnp"], ["ps", "-C", "mosquitto,java", "-o", "pid,ppid,comm,stat"]):
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=10)
    command = ["docker", "exec", "--user", "root", container,
        "/diag/lib/ld-linux-x86-64.so.2", "--library-path", "/diag/lib", "/diag/gdb",
        *debugger_arguments()]
    with output.open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=20)
    return {"exit": result.returncode, "scope": "owned candidate PID 1, after shutdown timeout", "pipeline_modified": False}
