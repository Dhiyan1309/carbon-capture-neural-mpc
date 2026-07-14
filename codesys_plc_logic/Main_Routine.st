PROGRAM PLC_PRG
VAR
    AI_Network_Input AT %IW0 : INT; 
    Physical_Valve : INT;           
    Max_Safe_Valve_Limit : INT := 80;
    Min_Safe_Valve_Limit : INT := 10;
END_VAR

Physical_Valve := AI_Network_Input;

IF Physical_Valve > Max_Safe_Valve_Limit THEN
    Physical_Valve := Max_Safe_Valve_Limit;
ELSIF Physical_Valve < Min_Safe_Valve_Limit THEN
    Physical_Valve := Min_Safe_Valve_Limit;
END_IF;
