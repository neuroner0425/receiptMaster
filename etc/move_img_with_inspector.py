import os
import random
import shutil
import sys
from PIL import Image
import matplotlib.pyplot as plt

form_dir = os.path.join('..', 'receipt_data', 'crawl')
to_dir = os.path.join('..', 'receipt_data', 'train', 'img')

from_files = [f for f in os.listdir(form_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print("'-' : 삭제 | '=' : 복구 | '`' : 프로그램 종료 | 기타 키 : 건너뜀 (창에서 바로 키 입력, 항상 전체화면)\n")

for file in from_files:
    file_path = os.path.join(form_dir, file)

    user_action = {"key": None}

    def on_key(event):
        if event.key == '-':
            user_action["key"] = '-'
            plt.close()
        elif event.key == '=':
            user_action["key"] = '='
            plt.close()
        else:
            user_action["key"] = 'skip'
            plt.close()

    fig = plt.figure()
    img = Image.open(file_path)
    plt.imshow(img)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', on_key)

    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except AttributeError:
        try:
            manager.window.state('zoomed')
        except Exception:
            pass

    plt.show()   # 키 누르면 plt.close()됨

    key = user_action["key"]
    if key == '-':
        os.remove(file_path)
        print(f"{file} 삭제됨.\n")
    elif key == '=':
        current_count = len([
            f for f in os.listdir(to_dir)
            if os.path.isfile(os.path.join(to_dir, f))
        ])
        dst_path = os.path.join(to_dir, f'{current_count}.{file.split(".")[-1]}')
        shutil.move(file_path, dst_path)
    else:
        print("프로그램을 종료합니다.")
        sys.exit(0)