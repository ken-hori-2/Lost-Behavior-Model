
import math
from reference_match_rate_Robosin import Property
import copy
import pprint
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson

class Algorithm_bp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 8 # 10 # 4
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.TRIGAR_REVERSE = False
        self.BACK = False
        self.BACK_REVERSE = False
        self.on_the_way = False
        self.bf = True
        self.STATE_HISTORY = []
        self.BPLIST = []
        self.PROB = []
        self.Arc = []
        self.OBS = []
        self.Storage_Arc = []
        self.SAVE = []

        "============================================== Visualization ver. との違い =============================================="
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"]
        # self.COST = move_cost_cal()
        self.backed = []
        self.Unbacked = self.Node_l
        "============================================== Visualization ver. との違い =============================================="
        self.n_m = arg[5]
            

    def BP(self, STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, total_stress, SAVE_ARC, TOTAL_STRESS_LIST, move_cost_result, test_bp_st_pre, move_cost_result_X, standard_list, rate_list):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        self.OBS = OBS
        self.BPLIST = BPLIST
        self.Advance_action = action
        self.bf = True
        self.state_history_first = True
        self.Add_Advance = Add_Advance
        "Add 1215"
        self.Backed_just_before = False
        self.total_stress = total_stress
        self.SAVE_ARC = SAVE_ARC
        self.first_pop = True
        self.BackPosition_finish = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        "============================================== Visualization ver. との違い =============================================="
        self.move_cost_result_pre = move_cost_result # self.l
        self.test_bp_st_pre = test_bp_st_pre
        # self.BPLIST = pd.Series(self.BPLIST, index=self.Node_l)
        self.test_bp_st = copy.copy(self.test_bp_st_pre)
        print("===== Storage : ", self.test_bp_st)
        X = self.Node_l.index("x") # self.new)
        self.move_cost_result = move_cost_result_X # shortest_path(self.move_cost_result_pre, indices=X, directed=False) # bpで使う
        self.move_cost_result_copy = copy.deepcopy(self.move_cost_result)
        "============================================== Visualization ver. との違い =============================================="
        
        while not self.done:

            "============================================== Visualization ver. との違い =============================================="
            "戻る行動の可視化ver.の場合はここにReverseが入る"
            "============================================== Visualization ver. との違い =============================================="

            if self.BACK or self.bf:
                    try:
                        
                        if self.bf: # ストレスが溜まってから初回
                            self.w = self.OBS
                            print(f"🥌 WEIGHT = {self.w}")
                            print("SAVE ARC : {}".format(self.SAVE_ARC))

                            if self.Add_Advance:

                                # ユークリッド距離
                                self.Arc = [math.sqrt((self.BPLIST[-1].row - self.BPLIST[x].row) ** 2 + (self.BPLIST[-1].column - self.BPLIST[x].column) ** 2) for x in range(len(self.BPLIST))]

                                "----- move cost -----"
                                print("bp-----=========================================================================================\n")
                                print("mv_copy : ", self.move_cost_result_copy)
                                self.move_cost_result_copy[self.move_cost_result_copy == np.inf] = np.nan
                                print("mv_copy inf -> nan : ", self.move_cost_result_copy)
                                print(type(self.move_cost_result_copy))
                                print("-----")
                                self.move_cost_result_copy = pd.Series(self.move_cost_result_copy, index=self.Node_l) # index=self.Unbacked)
                                print("mv_copy -> pandas + add index: ", self.move_cost_result_copy)
                                print(type(self.move_cost_result_copy))
                                print("-----")
                                self.move_cost_result_copy.dropna(inplace=True)
                                print("mv_copy drop nan: ", self.move_cost_result_copy)
                                print(type(self.move_cost_result_copy))
                                print("-----")
                                self.move_cost_result_copy.drop(index=["x"], inplace=True)
                                print("mv_copy drop x: ", self.move_cost_result_copy)
                                # # print(self.move_cost_result_copy["s"])
                                # # self.move_cost_result_copy.drop(index=self.backed, columns=self.backed, inplace=True)
                                self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
                                print("mv_copy drop backed: ", self.move_cost_result_copy)


                                print(f"test_bp_st : \n{self.test_bp_st}")
                                self.test_bp_st.dropna(inplace=True)
                                print("test bp st drop nan : ", self.test_bp_st)
                                print("-----")
                                # self.test_bp_st.drop(index=["x"], inplace=True)
                                # print(self.test_bp_st)

                                self.test_bp_st.drop(index=self.backed, inplace=True)
                                print("Storage : ", self.test_bp_st)

                                print("bp-----=========================================================================================\n")
                                "----- move cost -----"

                            print(f"🥌 WEIGHT = {self.w}")
                            print("👟 Arc[移動コスト]:{}".format(self.move_cost_result_copy))
                            print("📂 Storage {}".format(self.test_bp_st))
                            "============================================== Visualization ver. との違い =============================================="  
                        else:
                            print(f"🥌 WEIGHT = {self.w}")
                            print("👟 Arc[移動コスト]:{}".format(self.move_cost_result_copy))
                            print("📂 Storage {}".format(self.test_bp_st))
                        self.bf = False
                        self.BACK = False
                        
                        "----- Add Move Cost -----"
                        self.BPLIST = self.test_bp_st
                        
                        self.next_position = self.agent.back_position(self.BPLIST, self.w, self.Arc, self.move_cost_result_copy)
                        
                        "============================================== Visualization ver. との違い =============================================="
                        print("===== TEST 1114 =====")
                        print(self.BPLIST)
                        print(self.next_position)
                        print(type(self.BPLIST))
                        print(type(self.next_position))
                        next_lm = (self.BPLIST[self.BPLIST == self.next_position])
                        print("next_lm : ", next_lm)
                        print("next_lm : ", next_lm.index[0])

                        "test index"
                        test_index = self.BPLIST.index.get_loc(next_lm.index[0])
                        print("self.BPLIST.index : ", self.BPLIST.index.get_loc(next_lm.index[0]))

                        next_cost = next_lm.index[0]
                        print("next cost : ", next_cost)

                        
                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:{self.next_position}\n==================================")
                        self.on_the_way = True 
                    # except:
                    except Exception as e:
                        print('=== エラー内容 ===')
                        print('type:' + str(type(e)))
                        print('args:' + str(e.args))
                        print('message:' + e.message)
                        print('e自身:' + str(e))
                        print("ERROR!")
                        print("リトライ行動終了！")
                        print(" = 戻り切った状態 🤖🔚")
                        self.BackPosition_finish = True
                        break
            try:

                if self.state == self.next_position:

                    self.Backed_just_before = True

                    __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
                    pprint.pprint(self.n_m)
                    print("__a = ", __a)
                    self.n = __a[0] # nを代入
                    self.M = __a[1] # mを代入
                    print(f"[n, m] = {self.n, self.M}")
                    self.phi = [self.n, self.M]
                    print("👍 (bp) phi = ", self.phi)
                    print("phi(%) = ", self.n/(self.n+self.M), 1-self.n/(self.n+self.M))

                    
                    self.backed.append(next_cost)
                    self.Unbacked = [i for i in Node if i not in self.backed]
                    print("----- Backed : ", self.backed)
                    print("----- UnBacked : ", self.Unbacked)
                    print("----- test_bp_st -----")
                    print("削除前")
                    print("Storage : ", self.test_bp_st)
                    print("削除後")
                    # self.test_bp_st.drop(index=next_lm.index, inplace=True)
                    "一旦printして可視化するためだけのもの"
                    self.test_bp_st_copy = copy.copy(self.test_bp_st_pre)
                    self.test_bp_st_copy.dropna(inplace=True)
                    self.test_bp_st_copy.drop(index=self.backed, inplace=True)
                    print("Storage : ", self.test_bp_st_copy)
                    print("----- test_bp_st -----")
                    print("----- move cost -----")
                    print("削除前")
                    print("mv_copy : ", self.move_cost_result_copy)
                    print("削除後")
                    
                    "Add 1116 一旦printして可視化するためだけのもの 今は下の方でやっているが、こっちで可視化用のcopy2でやってもいい"
                    self.move_cost_result_copy2 = copy.copy(self.move_cost_result)
                    self.move_cost_result_copy2 = pd.Series(self.move_cost_result_copy2, index=self.Node_l) # index=self.Unbacked)
                    self.move_cost_result_copy2.drop(index=self.backed, columns=self.backed, inplace=True)
                    self.move_cost_result_copy2[self.move_cost_result_copy2 == np.inf] = np.nan
                    self.move_cost_result_copy2.dropna(inplace=True)
                    self.move_cost_result_copy2.drop(index=["x"], inplace=True)
                    print("mv_copy2 : ", self.move_cost_result_copy2)
                    # print("mv_copy : ", self.move_cost_result_copy)
                    " ↑ or ↓ "
                    # "----- これだとinplace=Trueでも、この後アルゴリズムを抜けるのでこの結果はリセットされてしまい反映されない -----"
                    # self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
                    # "-> エラー 上でindexのdropを既に削除しているのでエラーになる"
                    # print("mv_copy drop backed: ", self.move_cost_result_copy)
                    # print("mv_copy : ", self.move_cost_result_copy)
                    "-----------------------------------------------------------------------------------------------"

                    print("----- move cost -----")

                    self.BPLIST, self.w, self.OBS = self.agent.back_end(self.BPLIST, self.next_position, self.w, self.OBS, test_index, self.move_cost_result)
                    self.BACK =True
                    print("🔚 ARRIVE AT BACK POSITION (戻り終わりました。)")
                    print(f"🤖 State:{self.state}")
                    print("OBS : {}".format(self.OBS))

                    # self.total_stress = 0
                    "⚠️ 要検討 ⚠️ 戻った時にどのくらい減少させるか test_s = 進んだ分だけ減少させるか = その場所までのストレスまで減少させるか"
                    print("⚠️ total : {}".format(self.total_stress))
                    delta_s = self.Observation[self.state.row][self.state.column]
                    delta_s = round(abs(1.0-delta_s), 3)
                    if delta_s > 2:
                        delta_s = 1.0
                    "1217 コメントアウト"
                    # if self.total_stress - delta_s >= 0:
                    #     self.total_stress -= delta_s
                    # else:
                    #     self.total_stress = 0

                    print("⚠️ delta_s : {}".format(delta_s))
                    print("⚠️ total : {}".format(self.total_stress))
                    "-- 追加部分 --"
                    "============================================== Visualization ver. との違い =============================================="

                    self.STATE_HISTORY.append(self.state)
                    self.TOTAL_STRESS_LIST.append(self.total_stress)

                    "基準距離, 割合の可視化"
                    self.test_s = 0
                    self.standard_list.append(self.test_s)
                    # self.rate_list.append(self.n/(self.M+self.n))    # ○
                    self.rate_list.append(self.M/(self.M+self.n))      # ×

                    self.TRIGAR = False
                    self.TRIGAR_REVERSE = False

                    print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                    break

                else:
                    if self.on_the_way:
                        self.on_the_way = False
                    else:
                        print("🔛 On the way BACK")
            # except:
            except Exception as e:
                    print('=== エラー内容 ===')
                    print('type:' + str(type(e)))
                    print('args:' + str(e.args))
                    print('message:' + e.message)
                    print('e自身:' + str(e))
                    print("state:{}".format(self.state))
                    print("これ以上戻れません。 終了します。")
                    break # expansion 無しの場合は何回も繰り返さない
                
            print(f"🤖 State:{self.state}")
            if not self.state_history_first:
                self.STATE_HISTORY.append(self.state)
                self.TOTAL_STRESS_LIST.append(self.total_stress)

                "基準距離, 割合の可視化"
                self.test_s = 0
                self.standard_list.append(self.test_s)
                # self.rate_list.append(self.n/(self.M+self.n))    # ○
                self.rate_list.append(self.M/(self.M+self.n))      # ×

            print(f"Total Stress:{self.total_stress}")
            print("TRIGAR : {}".format(self.TRIGAR))
            self.state_history_first = False
            self.state = self.next_position
            print("COUNT : {}".format(self.COUNT))
            if self.COUNT > 100:
                print("\n######## BREAK ########\n")
                # breakではなくて、戻る場所に戻れないから別の戻る場所にするとか
                print("\n📂 Storage {}\n\n\n".format(self.BPLIST))
                break
            self.COUNT += 1
        self.COUNT = 0

        return self.total_stress, self.STATE_HISTORY, self.state, self.OBS, self.BackPosition_finish, self.TOTAL_STRESS_LIST, self.move_cost_result_pre, self.test_bp_st_pre, self.Backed_just_before, self.standard_list, self.rate_list