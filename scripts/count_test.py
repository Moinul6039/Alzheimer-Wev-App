import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(root, 'dataset', 'test')
if not os.path.isdir(test_dir):
    print('No test dir', test_dir)
else:
    total=0
    for folder in sorted(os.listdir(test_dir)):
        p=os.path.join(test_dir,folder)
        if os.path.isdir(p):
            files=[f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))]
            print(folder, len(files))
            total+=len(files)
    print('TOTAL', total)
