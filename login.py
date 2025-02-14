import requests, json, time, yaml

CONFIG_FILE = "config.yml"
with open(CONFIG_FILE, "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.BaseLoader)

# User Variables
USERNAME = cfg["config"]["USERNAME"]
PASSWORD = cfg["config"]["PASSWORD"]
CLIENT_ID = cfg["config"]["CLIENT_ID"]
UUID = cfg["config"]["UUID"]

# Application Constants
RDP_version = cfg["config"]["RDP_version"]
base_URL = cfg["config"]["base_URL"]
category_URL = cfg["config"]["category_URL"]
endpoint_URL = cfg["config"]["endpoint_URL"]
CLIENT_SECRET = cfg["config"]["CLIENT_SECRET"]
TOKEN_FILE = cfg["config"]["TOKEN_FILE"]
SCOPE = cfg["config"]["SCOPE"]

TOKEN_ENDPOINT = base_URL + category_URL + RDP_version + endpoint_URL


def _requestNewToken(refreshToken):
    if refreshToken is None:
        tData = {
            "username": USERNAME,
            "password": PASSWORD,
            "grant_type": "password",
            "scope": SCOPE,
            "takeExclusiveSignOnControl": "true",
        }
    else:
        tData = {
            "refresh_token": refreshToken,
            "grant_type": "refresh_token",
        }

    # Make a REST call to get latest access token
    response = requests.post(
        TOKEN_ENDPOINT,
        headers={"Accept": "application/json"},
        data=tData,
        auth=(CLIENT_ID, CLIENT_SECRET),
    )

    if response.status_code != 200:
        raise Exception(
            "Failed to get access token {0} - {1}".format(
                response.status_code, response.text
            )
        )

    # Return the new token
    return json.loads(response.text)


def changePassword(user, oldPass, clientID, newPass):
    tData = {
        "username": user,
        "password": oldPass,
        "grant_type": "password",
        "scope": SCOPE,
        "takeExclusiveSignOnControl": "true",
        "newPassword": newPass,
    }

    # Make a REST call to get latest access token
    response = requests.post(
        TOKEN_ENDPOINT,
        headers={"Accept": "application/json"},
        data=tData,
        auth=(clientID, CLIENT_SECRET),
    )

    if response.status_code != 200:
        raise Exception(
            "Failed to change password {0} - {1}".format(
                response.status_code, response.text
            )
        )

    tknObject = json.loads(response.text)
    # Persist this token for future queries
    saveToken(tknObject)
    # Return access token
    return tknObject["access_token"]


def saveToken(tknObject):
    tf = open(TOKEN_FILE, "w")
    # print("Saving the new token")
    # Append the expiry time to token
    tknObject["expiry_tm"] = time.time() + int(tknObject["expires_in"]) - 10
    # Store it in the file
    json.dump(tknObject, tf, indent=4)


def getToken():
    try:
        # print("Reading the token from: " + TOKEN_FILE)
        # Read the token from a file
        tf = open(TOKEN_FILE, "r+")
        tknObject = json.load(tf)

        # Is access token valid
        if tknObject["expiry_tm"] > time.time():
            # return access token
            return tknObject["access_token"]

        # print("Token expired, refreshing a new one...")
        tf.close()
        # Get a new token from refresh token
        tknObject = _requestNewToken(tknObject["refresh_token"])

    except Exception as exp:
        print("Caught exception: " + str(exp))
        print("Getting a new token using Password Grant...")
        tknObject = _requestNewToken(None)

    # Persist this token for future queries
    saveToken(tknObject)
    # Return access token
    return tknObject["access_token"]
