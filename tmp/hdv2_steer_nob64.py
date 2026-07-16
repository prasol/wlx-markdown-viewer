import requests
url='https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/hyperdiscovery-temp-20260716/tmp/hdv2_steer.py'
s=requests.get(url,timeout=60).text
old="print('STEERING_B64_BEGIN');b=base64.b64encode(archive.read_bytes()).decode();[print(b[i:i+8000]) for i in range(0,len(b),8000)];print('STEERING_B64_END')"
s=s.replace(old,"print('STEERING_DONE',flush=True)")
exec(compile(s,'hdv2_steer_nob64.py','exec'))
