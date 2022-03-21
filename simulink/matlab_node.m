Episode_time = 15;
rosinit()

while true
    start = rossubscriber('start_simulation');
    ini = receive(start);
    ini = ini.Data;
    if (ini == true)
      disp('Starting simulation...')
      sim('simulink_nodes',Episode_time);
     else
        disp('No starting recieved from python')
    end
end

