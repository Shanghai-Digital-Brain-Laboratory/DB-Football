# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#/bin/bash!

#this is to quickly render match up,  put the football_ai_light.py in the directory of gfootball /gfootball/env/players/ and run the following command

base_dir="$(dirname "$(dirname $PWD)")"
echo $base_dir
export PYTHONPATH=$base_dir

start=`date +%s`

ai_path='light_malib/trained_models/gr_football/11_vs_11/current_best'
ai_path2='light_malib/trained_models/gr_football/11_vs_11/defensive_passer'

ai_path="$base_dir/$ai_path"
ai_path2="$base_dir/$ai_path2"


#11v11
#python3 -m gfootball.play_game --players "football_ai_light:left_players=10,checkpoint=${ai_path};football_ai_light:right_players=10,checkpoint=${ai_path2}" --action_set=default --level "10_vs_10_kaggle" --real_time=false
python3 -m gfootball.play_game --players "football_ai_light:left_players=10,checkpoint=${ai_path}" --action_set=default --level "10_vs_10_kaggle" --real_time=false
#python3 -m gfootball.play_game --players "keyboard:left_players=1;football_ai_light:right_players=10,checkpoint=${ai_path}" --action_set=full --level "10_vs_10_kaggle" --real_time=true
#python3 -m gfootball.play_game --players "keyboard:left_players=1;football_ai_light:left_players=9,checkpoint=${ai_path};football_ai:right_players=10,checkpoint=${ai_path2}" --action_set=full --level "10_vs_10_kaggle" --real_time=true


end=`date +%s`
echo Execution time was `expr $end - $start` seconds.