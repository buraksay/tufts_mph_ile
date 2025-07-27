import xgboost

build_info = xgboost.build_info()
for name in sorted(build_info.keys()):
    print(f'{name}: {build_info[name]}')
