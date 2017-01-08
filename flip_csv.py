

f_in = open('driving_log.csv','r')
flipped_file = open('flipped_driving_log.csv','w+')
out_string = ""
line = f_in.readline()
while True:
    line = f_in.readline()
    print(line.split(","))
    if not line:
        break
    center,left,right,steering,throttle,brake,speed = line.split(",")
    steering = str(0-float(steering) )
    center=center.replace("IMG/","IMG/flip_")
    right=right.replace("IMG/","IMG/flip_")
    left=left.replace("IMG/","IMG/flip_")
    out_string += center +', '+ right +', '+ left  +', '+ steering +', '+ throttle +', '+ brake +', '+ speed 

print(out_string)

f_in.close
flipped_file.write(out_string)
