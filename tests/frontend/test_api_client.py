import io,json,unittest
from unittest.mock import patch
from urllib.error import HTTPError,URLError
from services.frontend.api_client import ApiClient,ApiConnectionError,ApiNotFoundError,ApiValidationError

class Response:
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def read(self):return json.dumps(self.value).encode()
class ApiClientTests(unittest.TestCase):
    def test_typed_methods(self):
        with patch("services.frontend.api_client.urlopen",return_value=Response([{"id":"p1"}])) as call:
            self.assertEqual(ApiClient().get_persons()[0]["id"],"p1");self.assertIn("/persons",call.call_args.args[0].full_url)
    def test_not_found_mapping(self):
        error=HTTPError("x",404,"missing",{},io.BytesIO(b'{"detail":"missing"}'))
        with patch("services.frontend.api_client.urlopen",side_effect=error),self.assertRaises(ApiNotFoundError):ApiClient().get_person("x")
    def test_connection_mapping(self):
        with patch("services.frontend.api_client.urlopen",side_effect=URLError("down")),self.assertRaises(ApiConnectionError):ApiClient().get_health()
    def test_malformed_person_list_is_not_accepted(self):
        with patch("services.frontend.api_client.urlopen",return_value=Response({"items":[]})),self.assertRaises(ApiValidationError):ApiClient().get_persons()
    def test_camera_list_contract(self):
        with patch("services.frontend.api_client.urlopen",return_value=Response([{"id":"CAM-X","enabled":True}])):self.assertEqual(ApiClient().get_cameras()[0]["id"],"CAM-X")
        with patch("services.frontend.api_client.urlopen",return_value=Response([{}])),self.assertRaises(ApiValidationError):ApiClient().get_cameras()
