roscore &
gnome-terminal -- cd rl_prj &
rosrun rl_connection agent_node.py &
gnome-terminal -- --logdir ./gradient_tape &
gnome-terminal -- matlab -nodesktop -r "run('./simulink/matlab_node.m')" &




