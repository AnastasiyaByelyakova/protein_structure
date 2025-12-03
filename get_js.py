
import os
import urllib.request

os.makedirs("static/js", exist_ok=True)

# Fetch 3Dmol.js from CDN
v3dmol_url = "https://3dmol.csb.pitt.edu/build/3Dmol.js"
with open("static/js/3Dmol.js", "w") as f:
    f.write(urllib.request.urlopen(v3dmol_url).read().decode())

# Fetch jquery from CDN
jquery_url = "https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"
with open("static/js/jquery.min.js", "w") as f:
    f.write(urllib.request.urlopen(jquery_url).read().decode())

print("Successfully wrote 3Dmol.js and jquery.min.js")
