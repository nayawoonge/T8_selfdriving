import threading

# 전역 변수 선언
keys_pressed = {
    'w': False,
    's': False,
    'a': False,
    'd': False,
    'e': False,
    'k': False,
    'z': False
}

def key_input_thread():
    while True:
        char = getch()
        if char in keys_pressed:
            keys_pressed[char] = True
        elif char == 'q':  # 'q' 키를 누르면 프로그램 종료
            break

# 별도의 스레드에서 키 입력을 처리
thread = threading.Thread(target=key_input_thread)
thread.start()

try:
    while True:
        if keys_pressed['w']:
            # Forward 코드...
        if keys_pressed['s']:
            # Backward 코드...
        if keys_pressed['a']:
            # Left 코드...
        if keys_pressed['d']:
            # Right 코드...
        # ... 나머지 코드...

except KeyboardInterrupt:
    pass

finally:
    # 스레드를 종료하기 위해 'q'를 설정
    keys_pressed['q'] = True
    thread.join()
