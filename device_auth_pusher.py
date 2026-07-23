import os
import time
import base64
import requests
import sys

# Official GitHub CLI OAuth App Client ID
CLIENT_ID = "178c1a00c2e32a30949d"

def start_device_flow():
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "client_id": CLIENT_ID,
        "scope": "repo workflow user"
    }

    print("[GitHub Auth] Requesting device authorization code...")
    res = requests.post("https://github.com/login/device/code", headers=headers, json=payload)
    if res.status_code != 200:
        print(f"[ERROR] Failed to start device authorization flow: {res.text}")
        return None

    data = res.json()
    user_code = data["user_code"]
    verification_uri = data["verification_uri"]
    device_code = data["device_code"]
    interval = data.get("interval", 5)

    print("\n" + "=" * 80)
    print("🔑 GITHUB DEVICE AUTHORIZATION REQUIRED")
    print(f"👉 1. Open URL in your browser: {verification_uri}")
    print(f"👉 2. Enter User Code        : {user_code}")
    print("=" * 80 + "\n")
    print(f"Waiting for user authorization at {verification_uri}...")

    # Poll for access token
    poll_url = "https://github.com/login/oauth/access_token"
    poll_payload = {
        "client_id": CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
    }

    while True:
        time.sleep(interval)
        poll_res = requests.post(poll_url, headers=headers, data=poll_payload)
        poll_data = poll_res.json()

        if "access_token" in poll_data:
            access_token = poll_data["access_token"]
            print("\n[SUCCESS] Device authorized successfully!")
            return access_token
        
        error = poll_data.get("error")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            interval += 5
            print(".", end="", flush=True)
            continue
        else:
            print(f"\n[ERROR] Authorization failed: {error}")
            return None

def upload_all_files(token: str, repo_name: str = "optimisation-only-dummy"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get username
    user_res = requests.get("https://api.github.com/user", headers=headers)
    if user_res.status_code != 200:
        print(f"[ERROR] Could not fetch user details: {user_res.text}")
        return False

    username = user_res.json()["login"]
    print(f"[OK] Logged in as GitHub User: {username}")

    # Ensure repository exists
    repo_url = f"https://api.github.com/repos/{username}/{repo_name}"
    repo_check = requests.get(repo_url, headers=headers)

    if repo_check.status_code == 404:
        print(f"[GitHub] Creating repository '{repo_name}' for user {username}...")
        create_res = requests.post("https://api.github.com/user/repos", headers=headers, json={"name": repo_name, "private": False})
        if create_res.status_code not in [200, 201]:
            print(f"[ERROR] Could not create repository: {create_res.text}")
            return False

    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    master_report_path = r"C:\Users\23aiml29\.gemini\antigravity-ide\brain\33c5932e-261b-4f93-85ad-8a65e169078a\master_report.md"

    files_to_upload = []

    if os.path.exists(master_report_path):
        with open(master_report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        files_to_upload.append(("README.md", report_content.encode("utf-8")))
        files_to_upload.append(("master_report.md", report_content.encode("utf-8")))

    ignore_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".gemini", "cache"}
    for root, dirs, files in os.walk(workspace_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if file in [".DS_Store", "device_auth_pusher.py", "github_pusher.py"] or file.endswith(".pyc"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, workspace_dir).replace("\\", "/")
            if os.path.getsize(full_path) > 50 * 1024 * 1024:
                continue

            with open(full_path, "rb") as f:
                content = f.read()
            files_to_upload.append((rel_path, content))

    print(f"\n[GitHub] Uploading {len(files_to_upload)} files to https://github.com/{username}/{repo_name}...")

    success = 0
    for rel_path, content_bytes in files_to_upload:
        encoded_content = base64.b64encode(content_bytes).decode("utf-8")
        file_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{rel_path}"

        get_res = requests.get(file_url, headers=headers)
        sha = get_res.json().get("sha") if get_res.status_code == 200 else None

        put_payload = {"message": f"Upload {rel_path} via Antigravity Suite", "content": encoded_content}
        if sha:
            put_payload["sha"] = sha

        put_res = requests.put(file_url, headers=headers, json=put_payload)
        if put_res.status_code in [200, 201]:
            print(f"  • Uploaded: {rel_path}")
            success += 1
        else:
            print(f"  [FAIL] Failed: {rel_path} ({put_res.status_code})")

    print("\n" + "=" * 80)
    print(f"🎉 COMPLETE! Uploaded {success}/{len(files_to_upload)} files to:")
    print(f"👉 https://github.com/{username}/{repo_name}")
    print("=" * 80)
    return True

if __name__ == "__main__":
    token = start_device_flow()
    if token:
        upload_all_files(token)
