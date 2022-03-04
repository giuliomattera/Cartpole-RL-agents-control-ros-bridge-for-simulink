roscore &
gnome-terminal -- matlab -nodesktop -r "run('./simulink/matlab_node.m')" &
gnome-terminal -- rosrun rl_connection agent_node.py &


