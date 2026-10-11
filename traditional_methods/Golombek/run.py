import os, glob, subprocess, shutil

# FILE_NAMES = ['TRA_000846_2475']

FILE_NAMES = ['ESP_016526_2415', 'ESP_049182_1285', 'PSP_001391_2465', 'TRA_000828_2495', 'TRA_000841_2460', 'TRA_000894_2475']
# FILE_NAMES = ['PSP_005775_1245', 'ESP_062954_1550', 'ESP_088830_2090', 'ESP_038943_2040', 'ESP_011625_1275', 'ESP_027411_1370']

for fn in FILE_NAMES:
    exe = r"F:\mars_rock\code\2008\ROSCO-main\RockDetector\build\bin\Release\ExportRocksPGM.exe"
    dir_path = r"F:\mars_rock\unsupervised_mars_rock_detection\benchmark_methods\Golombek_2008\train\{}\pgm".format(fn)
    params = os.path.join(dir_path, "{}_params.txt".format(fn))

    rocks_dir = os.path.join(dir_path, "Rocks_output")
    outs_dir = os.path.join(dir_path, "Outs_output")
    os.makedirs(rocks_dir, exist_ok=True)
    os.makedirs(outs_dir, exist_ok=True)

    pgms = sorted(glob.glob(os.path.join(dir_path, "*.pgm")))
    pgms = [p for p in pgms if not os.path.basename(p).startswith(("Rocks-", "Shads-", "Outs-"))]

    for p in pgms:
        base = os.path.splitext(os.path.basename(p))[0]
        cmd = [exe, "-i", p, "-p", params, "-o", base]
        r = subprocess.run(cmd, cwd=dir_path, capture_output=True, text=True)

        if r.returncode != 0:
            print("ERROR:", base, r.stderr.strip() or r.stdout.strip())
            continue

        rocks_fp = os.path.join(dir_path, f"Rocks-{base}.pgm")
        if os.path.exists(rocks_fp):
            shutil.move(rocks_fp, os.path.join(rocks_dir, os.path.basename(rocks_fp)))

        outs_fp = os.path.join(dir_path, f"Outs-{base}.pgm")
        if os.path.exists(outs_fp):
            shutil.move(outs_fp, os.path.join(outs_dir, os.path.basename(outs_fp)))

        shads_fp = os.path.join(dir_path, f"Shads-{base}.pgm")
        if os.path.exists(shads_fp):
            os.remove(shads_fp)

        tmp_candidates = [
            os.path.join(dir_path, "shadow_tmp.pgm"),
            os.path.join(dir_path, f"{base}.__shadow_tmp.pgm"),
            os.path.join(dir_path, f"{base}_shadow_tmp.pgm"),
        ]
        for tp in tmp_candidates:
            if os.path.exists(tp):
                os.remove(tp)

    print("Done:", fn)