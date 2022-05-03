Episode_time = 10;
TS = 1e-3;
x_limit = 1.5;
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

