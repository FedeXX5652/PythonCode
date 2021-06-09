import keyboard

shot_pressed = 0
was_pressed = False

while True:
    if keyboard.is_pressed('s'):
        if not was_pressed:
            shot_pressed += 1
            print("shot_pressed %d times"%shot_pressed)
            was_pressed = True
            
    else:
        was_pressed = False