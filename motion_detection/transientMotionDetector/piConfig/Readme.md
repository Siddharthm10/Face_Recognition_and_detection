# Configuration on a Raspberry pi
1. Flash the SD card with Raspbian lite and install it on the Raspberry Pi.
2. Connect the raspberry pi to the network via an ethernet cable.
3. Check the IP of the raspberry pi via your router config.
4. Install the ssh library(for connecting to devices on your network):   
    `sudo apt-get update` <br>
    `sudo apt-get upgrade` <br>
    `sudo apt install openssh-server`
5. Enter the following command to connect to raspberry pi:  
      `ssh <Username>@<IP_address> `   
    Example:-  
      `ssh pi@192.16.3.510 `

6. Install OpenCV:  
    `pip install opencv-python`

7. Download the files using :  
    `git clone `

    OR  

   Copy the files from dekstop to Raspberry Pi: 
    In other terminal:  

    `scp -r ./motion_detection/transientMotionDetector/  pi@192.16.3.510:~`

8. Run the script with :  
     `python3 pi_m_detector`

9.  Copy the data back to the localhost to check the output  
    `scp -r pi@192.16.3.510:~transientMotionDetector/piConfig/ ./motion_detection/transientMotionDetector/  `

10. You will get your output in the given directory.