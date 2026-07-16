import sys, requests
url='https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/hyperdiscovery-temp-20260716/tmp/hdv2_analyze.py'
s=requests.get(url,timeout=60).text
s=s.replace("columns={{0:'accuracy'}}","columns={0:'accuracy'}")
exec(compile(s,'hdv2_analyze_fixed.py','exec'))
