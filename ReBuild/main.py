from enum import Enum
from pprint import pprint
from tkinter import FIRST
import numpy as np
import random
from sklearn import preprocessing
from environment_prob import Environment
from bp_Robosin import Algorithm_bp
from exp_Robosin import Algorithm_exp
from agent import Agent
from setting_large_grid_Robosin import Setting
import pprint
from advance_Robosin import Algorithm_advance
from reference_match_rate_Robosin import Property
import pandas as pd





def main():

    # Try 10 game.
    result = []
    little = []
    over = []
    goal_count = 0

    for test in range(1):

        set = Setting()

        NODELIST, ARCLIST, Observation, map, grid, n_m = set.Infomation()
        test = [grid, map, NODELIST]
        marking_param = 1
        env = Environment(*test)
        agent = Agent(env, marking_param, *test)
        STATE_HISTORY = []
        CrossRoad = []
        TOTAL_STRESS_LIST = []
        move_cost_result = []
        standard_list = []
        rate_list = []
        import numpy as np
        test_bp_st = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        
        for x in range(1):
            print("=================== {} Steps\n===================".format(x))
            
            # Initialize position of agent.
            state = env.reset()
            demo = [state, env, agent, NODELIST, Observation, n_m] # , reference]
            Advance_action = Algorithm_advance(*demo)
            back_position = Algorithm_bp(*demo)
            explore_action = Algorithm_exp(*demo)
            TRIGAR = False
            OBS = []
            total_stress = 0
            move_step = 0
            old_to_adavance = "s"
            Backed_just_before = False
            phi = [1, 1] # n, m
            test_s = 0
            
            for i in range(20): # 4 戻るノードの個数以上は回す

                total_stress, STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, GOAL, SAVE_ARC, CrossRoad, Storage, Storage_Stress, TOTAL_STRESS_LIST, move_cost_result, test_bp_st, move_cost_result_X, standard_list, rate_list = Advance_action.Advance(STATE_HISTORY, state, TRIGAR, OBS, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, move_step, old_to_adavance, move_cost_result, test_bp_st, Backed_just_before, phi, standard_list, rate_list, test_s)
                if GOAL:
                    print("探索済みのノード Storage : {}".format(Storage))
                    print("未探索のノード CrossRoad : {}".format(CrossRoad))
                    print("探索終了")
                    break

                print("\n============================\n🤖 🔛　アルゴリズム切り替え -> agent Back position\n============================")

                total_stress, STATE_HISTORY, state, OBS, BackPosition_finish, TOTAL_STRESS_LIST, move_cost_result, test_bp_st, Backed_just_before, standard_list, rate_list = back_position.BP(STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, total_stress, SAVE_ARC, TOTAL_STRESS_LIST, move_cost_result, test_bp_st, move_cost_result_X, standard_list, rate_list)

                if BackPosition_finish:
                    BackPosition_finish = False
                    print(" = 戻り切った状態 🤖🔚 {}回目".format(x+1))
                    
                    "== map, bplist reset =="
                    "== Storageを空にするバージョン =="
                    set = Setting()
                    map = set.reset()
                    test = [grid, map, NODELIST] # , GOAL_STATE]
                    env = Environment(*test)
                    # agent = Agent(env, *test)
                    # marking_param += 1
                    agent = Agent(env, marking_param, *test)
                    break
                    "== Storageを空にするバージョン =="

                print("\n============================\n🤖 🔛　アルゴリズム切り替え -> agent Explore\n============================")

                total_stress, STATE_HISTORY, state, TRIGAR, CrossRoad, GOAL, TOTAL_STRESS_LIST, move_step, old_to_adavance, phi, standard_list, rate_list, test_s = explore_action.Explore(STATE_HISTORY, state, TRIGAR, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, Backed_just_before, standard_list, rate_list)

                if GOAL:
                    print("探索済みのノード Storage : {}".format(Storage))
                    print("未探索のノード CrossRoad : {}".format(CrossRoad))
                    print("探索終了")
                    break

                print("\n============================\n🤖 🔛　アルゴリズム切り替え -> agent Advance\n============================")

            print("Episode {}: Agent gets {} stress.".format(i, total_stress))
            print("STATE_HISTORY = {}".format(STATE_HISTORY))
            print("stress = {}".format(TOTAL_STRESS_LIST))
            print("standard_list = ", standard_list)
            print("rate_list = ", rate_list)
            print(len(TOTAL_STRESS_LIST))

            if GOAL:
                print(" {} 回目".format(x+1))
                print("retry : {} 回目".format(x))
                goal_count += 1
                break

        "======== データの出力 ========"
        df = pd.Series(data=STATE_HISTORY)
        df = df[df != df.shift(1)]
        print('-----削除後データ----')
        print("Steps:{}".format((len(df))))
        result.append((len(df)))
        if len(df) <= 100:
            little.append(len(df))

        if len(df) >= 1000:
            over.append(len(df))

    print(result)
    # print(result/100)
    print("最小:{}".format(min(result)))
    print("最大:{}".format(max(result)))
    print("150 以内 : {}".format(len(little)))
    print("1000以上 : {}".format(len(over)))
    print("goal : {}".format(goal_count))
    print("x(retry), i : {}, {}".format(x, i))
    Length_history = len(STATE_HISTORY)
    print("length State history: {}".format(Length_history))
    print("length Storage Stress : {}".format(len(TOTAL_STRESS_LIST)))

    # print("standard_list = ", standard_list)
    # print("rate_list = ", rate_list)


if __name__ == "__main__":
    main()
