import sys,tty,termios
import time, motoron
import select

mc = motoron.MotoronI2C(address=18, bus=8)
sm = motoron.MotoronI2C(address=16, bus=8)


# Reset the controller to its default settings, then disable CRC.  The bytes for
# each of these commands are shown here in case you want to implement them on
# your own without using the library.
mc.reinitialize()  # Bytes: 0x96 0x74
mc.disable_crc()   # Bytes: 0x8B 0x04 0x7B 0x43
sm.reinitialize() 
sm.disable_crc()
# Clear the reset flag, which is set after the controller reinitializes and
# counts as an error.
mc.clear_reset_flag()  # Bytes: 0xA9 0x00 0x04
sm.clear_reset_flag()


# Configure motor 1
mc.set_max_acceleration(1, 0)
mc.set_max_deceleration(1, 0)

# Configure motor 2
mc.set_max_acceleration(2, 0)
mc.set_max_deceleration(2, 0)
# Configure steering motor/ motor 3
sm.set_max_acceleration(1, 0)
sm.set_max_deceleration(1, 0)

vel = 0
#counter for controlling smooth velocity reduction when steering kekw
count = 0
#counter for smooth steering control
count_steer = 0
count_neutral_steer = 0
max = 200
min = -200
steer = 0
steer_max = 200
steer_min = -200

def is_input_available():
    dude, [], [] = select.select([sys.stdin], [], [], 1)
    if(len(dude) == 0):
        return False
    else: 
        return True

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        if(is_input_available()):
            ch = sys.stdin.read(1)
        else:
            ch = "z"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch



# Instructions for when the user has an interface
print("w/s: acceleration")
print("a/d: steering")

def brake():
    global vel
    if(vel > 20):
        vel -= 20
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)    
    elif(vel < -20):
        vel += 20
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)
    else: 
        vel = 0

def backward():
    global vel
    if (vel < 0):
        neutral()
    #For instant motor activation upon press
    if (vel > 0 and vel < 40):
        vel = 40
    if(vel < max):
        vel += 1
    mc.set_speed(1, vel)
    mc.set_speed(2, vel)

def forward():
    global vel
    if(vel > 0):
        neutral()
    #For instant motor activation upon press
    if (vel < 0 and vel > -40):
        vel = -40
    if(vel > min):
        vel -= 1
    mc.set_speed(1, vel)
    mc.set_speed(2, vel)

def neutral():
    global vel
    if(vel > 10):
        vel -= 10
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)    
    elif(vel < -10):
        vel += 10
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)
    else: 
        vel = 0


def neutral_steer():
    global vel
    global count
    if(vel > 0):
        count += 1
        if(count > 15):
            vel -= 1
            count = 0
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)    
    elif(vel < 0):
        count += 1
        if(count > 15):
            vel += 1
            count = 0
        mc.set_speed(1, vel)
        mc.set_speed(2, vel)
    else: 
        vel = 0

# Infinite loop that will not end until the user presses the
# exit key
try:

    while True:
        # Keyboard character retrieval method is called and saved
        # into variable
        char = getch()

        # The car will drive forward when the "w" key is pressed
        if(char == "w"):
            #print("Forward")
            forward()
        # The car will reverse when the "s" key is pressed
        elif(char == "s"):
            #print("Backward")
            backward()

        # The "a" key will toggle the steering left
        elif(char == "a"):
            #print("Left")
            #global count_steer
            count_steer += 1
            if(count_steer > 10):
                if(steer > 0):
                    steer = 0
                    continue
                if(steer > steer_min):
                    steer -= 50
                count_steer = 0
                
            sm.set_speed(1, steer)
            neutral_steer()

        # The "d" key will toggle the steering right
        elif(char == "d"):
            #print("Right")
            count_steer += 1
            if(count_steer > 10):
                if(steer < 0):
                    steer = 0 
                    pass
                if (steer < steer_max):
                    steer += 50
                count_steer = 0

            sm.set_speed(1, steer)
            neutral_steer()

        elif(char == "e"):
            #print("Brake")
            brake()
            
        elif(char == "k"):
            #print("NAGA")
            break
        elif(char == "z"):
            neutral()
            
        print("vel: ", vel*(-1))
        print("steer: ", steer)

        


            
        
except KeyboardInterrupt:
    pass