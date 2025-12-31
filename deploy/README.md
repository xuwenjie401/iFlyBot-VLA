# server

examples
python deploy/server_policy.py --config_json deploy/tools/settings/state_lap_fast_joint.json --infer_json deploy/tools/settings/state_lap_fast_joint_infer.json

python deploy/server_policy.py --config_json deploy/tools/settings/libero_goal.json --infer_json deploy/tools/settings/libero_goal_infer.json


# client

python deploy/lindenbot/linden_offline_client_joint.py

python deploy/libero/libero_client.py --env libero_goal
