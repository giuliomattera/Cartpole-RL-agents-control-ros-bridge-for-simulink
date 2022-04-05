Episode_time = 4;
TS = 1e-3;
rosinit()
model = 'simulink_nodes';
while true
    start = rossubscriber('start_simulation');
    ini = receive(start);
    ini = ini.Data;
    if (ini == true)
      disp('Starting simulation...')
      sim(model);
     else
        disp('No starting recieved from python')
    end
end

