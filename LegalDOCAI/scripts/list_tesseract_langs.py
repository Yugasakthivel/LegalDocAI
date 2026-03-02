import subprocess, os, sys
def main():
    try:
        exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True)
        print("returncode:", out.returncode)
        print("stdout:\n", out.stdout)
        print("stderr:\n", out.stderr)
    except Exception as e:
        print("error:", e)
if __name__ == "__main__":
    main()
