import urllib.request, sys
URL = "https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/04ed00b22ef3b19eb0faa3fb62a56026a66fbcf2/experiments/hypergame_g1_hf_optimized_runner.py"
runner = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
old = "pars(g,ood)['max']//3]))))"
new = "pars(g,ood)['max']//3))))"
if old not in runner:
    raise RuntimeError("Expected typo patch point not found")
runner = runner.replace(old, new, 1)
sys.argv = ["hypergame_g1_hf_optimized_runner.py"] + sys.argv[1:]
exec(compile(runner, "hypergame_g1_hf_optimized_runner.py", "exec"), {"__name__": "__main__", "__file__": "hypergame_g1_hf_optimized_runner.py"})
