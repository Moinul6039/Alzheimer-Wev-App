import os, datetime
p = 'model/alzheimer_model.pth'
if os.path.exists(p):
    s = os.path.getsize(p)
    m = datetime.datetime.fromtimestamp(os.path.getmtime(p))
    print(f"{p} exists, size_bytes={s}, modified={m.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print(f"{p} MISSING")
