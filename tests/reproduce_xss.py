import streamlit as st
from streamlit.testing.v1 import AppTest
import html

def test_xss_in_segment_name():
    at = AppTest.from_file("app.py", default_timeout=30)
    # We bypass the need for an actual image upload by setting session state directly
    at.run()

    malicious_name = "<img src=x onerror=alert(1)>"
    at.session_state.segments = {malicious_name: "M 0 0 L 10 10"}
    at.run()

    # In AppTest, at.markdown is a list of Markdown elements
    found_vulnerable = False
    for md in at.markdown:
        if malicious_name in md.value:
            found_vulnerable = True
            print(f"VULNERABILITY CONFIRMED: Malicious string found unescaped in markdown: {md.value}")

    # Check success message if we can trigger it
    # We can't easily trigger the success message without a full flow, but we can check if it exists in the code

    if not found_vulnerable:
        print("Vulnerability not found in markdown (it might already be escaped or not rendered).")

    return found_vulnerable

if __name__ == "__main__":
    try:
        test_xss_in_segment_name()
    except Exception as e:
        print(f"Could not run test: {e}")
