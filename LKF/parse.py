import numpy as np

lidar_array = []
truth_array = []
prevx = 0
prevy = 0
prevtime = 0
file = open("original_data.txt","r")
for line in file:
    temp = line.split()
    if temp[0] == "L":
        time = float(temp[3])
        posx = float(temp[1])
        posy = float(temp[2])
        lidar_array.append([time, posx, posy, (posx-prevx)/(time-prevtime), (posy-prevy)/(time-prevtime)])
        prevx = posx
        prevy = posy
        prevtime = time
        tpx = float(temp[4])
        tpy = float(temp[5])
        tvx = float(temp[6])
        tvy = float(temp[7])
        truth_array.append([tpx, tpy, tvx, tvy])
np.save("data.npy" , np.array(lidar_array))
np.save("truth.npy", np.array(truth_array))