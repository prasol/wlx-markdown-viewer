import urllib.request

URL = "https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/5a9dadd93e537135d2c5b7758fa36b6f5cedd771/experiments/hypergame_g1_hf_compact.py"
source = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
old = "dmo,pmo=op_metric(op,hold,cfg['op_eval'],800+fi,dev,True);orows.append("
new = "dmo,pmo=op_metric(op,hold,cfg['op_eval'],800+fi,dev,True);op=op.cpu();orows.append("
if old not in source:
    raise RuntimeError("Expected GPU-device patch point was not found")
source = source.replace(old, new, 1)
globals_dict = {
    "__name__": "__main__",
    "__file__": "hypergame_g1_hf_compact.py",
    "SOURCE_TEXT": source,
}
exec(compile(source, "hypergame_g1_hf_compact.py", "exec"), globals_dict)
