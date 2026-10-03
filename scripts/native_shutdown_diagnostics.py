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
thread_by_lwp = {}
for candidate_thread in threads:
    ptid = candidate_thread.ptid
    lwp = int(ptid[1]) if len(ptid) > 1 and ptid[1] else 0
    if lwp:
        thread_by_lwp[lwp] = candidate_thread

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

                    # Ubuntu's stripped libc may omit the pthread_mutex_t debug
                    # type.  On x86-64, RDI is the first integer/pointer
                    # argument and glibc's public ABI layout begins
                    # __lock, __count, __owner, __nusers, __kind.  Use that
                    # combination only as an explicitly labeled inference,
                    # never as a typed/confirmed owner claim.
                    try:
                        arch = frame.architecture().name()
                    except gdb.error:
                        arch = ""
                    if source == "register_rdi" and "x86-64" in arch and len(words) >= 5:
                        lock_word, count, owner, nusers, kind = words[:5]
                        owner_thread = thread_by_lwp.get(owner)
                        plausible = (lock_word in (1, 2) and 0 <= count <= 1024 and
                                     owner > 0 and 0 <= nusers <= 1024 and
                                     0 <= kind <= 4096 and owner_thread is not None)
                        if plausible:
                            waiter = thread
                            print("F5_MUTEX_OWNER_INFERRED_GLIBC_X86_64 waiter_thread=%s owner_thread=%s owner_lwp=%s owner_name=%s address=%s lock=%s count=%s nusers=%s kind=%s" %
                                  (waiter.num, owner_thread.num, owner,
                                   owner_thread.name, hex(address), lock_word,
                                   count, nusers, kind))
                            owner_thread.switch()
                            print(gdb.execute("bt", to_string=True))
                            waiter.switch()
                except gdb.error as error:
                    print("F5_MEMORY_UNAVAILABLE %s %s" % (source, error))

                # When libc pthread types are available, read the mutex owner
                # from the actual pthread_mutex_t layout instead of guessing
                # from raw register/memory words.  An owner is considered
                # confirmed only when this typed read succeeds AND a GDB thread
                # with the same LWP/TID exists in the inferior.
                try:
                    mutex_type = gdb.lookup_type("pthread_mutex_t").pointer()
                    mutex = gdb.Value(address).cast(mutex_type).dereference()
                    owner = int(mutex["__data"]["__owner"])
                    lock_word = int(mutex["__data"]["__lock"])
                    owner_threads = []
                    for candidate_thread in threads:
                        ptid = candidate_thread.ptid
                        lwp = int(ptid[1]) if len(ptid) > 1 and ptid[1] else 0
                        tid = int(ptid[2]) if len(ptid) > 2 and ptid[2] else 0
                        if owner > 0 and owner in (lwp, tid):
                            owner_threads.append(candidate_thread)
                    print("F5_MUTEX_TYPED candidate=%s address=%s lock=%s owner=%s owner_thread_matches=%s" %
                          (source, hex(address), lock_word, owner,
                           [t.num for t in owner_threads]))
                    if owner_threads:
                        waiter = thread
                        for owner_thread in owner_threads:
                            owner_thread.switch()
                            print("F5_MUTEX_OWNER_CONFIRMED waiter_thread=%s owner_thread=%s owner_ptid=%s address=%s" %
                                  (waiter.num, owner_thread.num, owner_thread.ptid, hex(address)))
                            print(gdb.execute("bt", to_string=True))
                        waiter.switch()
                except (gdb.error, KeyError, TypeError, ValueError) as error:
                    print("F5_MUTEX_TYPED_UNAVAILABLE candidate=%s address=%s error=%s" %
                          (source, hex(address), error))
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
        "-ex", "attach 1", "-ex", "info threads", "-ex", "thread apply all bt",
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
