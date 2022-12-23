from pprint import pprint
import numpy as np
from reference_match_rate_Robosin import Property
import pprint
import random
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson
from scipy.sparse import csr_matrix
import pandas as pd
import copy
from neural_relu import neural

class Algorithm_advance():
    
    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property() # arg[5]
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 2.0
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
        self.test_bp_st_pre = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.PROB = []
        self.Arc = []
        self.OBS = []
        self.FIRST = True
        self.SAVE_ARC = []
        self.Storage = []
        self.Storage_Stress = []
        self.Storage_Arc = []
        self.DEMO_LIST = []
        self.SIGMA_LIST = []
        self.sigma = 0
        self.test_s = 0
        self.data_node = []
        self.XnWn_list = []
        self.save_s = []
        self.save_s_all = []
        self.End_of_O = False
        self.standard_list = []
        self.rate_list = []
        self.n_m = arg[5]
        
        "============================================== Visualization ver. との違い =============================================="
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"]
        "-- init --"
        self.old = "s"
        self.l = {"s":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "A":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "B":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "C":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "D":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "E":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "F":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "O":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "g":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  "x":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        self.Node = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"]
        self.l = pd.DataFrame(self.l, index = pd.Index(self.Node))
        
        self.move_cost_result = []
        self.test_bp_st_pre = pd.Series(self.test_bp_st_pre, index=self.Node_l)
        "============================================== Visualization ver. との違い =============================================="

    def hierarchical_model_O(self, ΔS): # 良い状態では小さいずれは気にしない(でもそもそも距離のずれは気にする必要ないかも)

        "hierarchical_model_Xから移動"
        if self.End_of_O: # 直前までに○の連続が途切れていた場合は一旦リセット
            self.n=1      # resetで0ではなく、1 -> 1/(1+1)=0.5となる
            # self.nnn=1    # resetで0ではなく、1 -> 1/(1+1)=0.5となる
            self.End_of_O = False

        self.n += 1
        # self.nnn+=1
        
        "×の連続数は良い状態には用いないので、ここでリセットしても関係ないから大丈夫"
        self.M=1      # resetで0ではなく、1 -> 1/(1+1)=0.5となる
        # self.mmm=1    # resetで0ではなく、1 -> 1/(1+1)=0.5となる
        Wn = np.array([1, -0.1])
        print("重みWn [w1, w2] : ", Wn)
        model = neural(Wn)
        print(f"入力Xn[ΔS, n] : {ΔS}, {self.n}")

        "===== 何連続から良い状態とするか -> n-?で決定 ====="
        # neu_fire, XnWn = model.perceptron(np.array([ΔS, self.n-3]), B=0) # Relu関数 これがあるとないとではゴール到達率が違う defalt:n=0
        "今回は3連続で良い状態とした(n-1)"
        neu_fire, XnWn = model.perceptron(np.array([ΔS, self.n-1]), B=0) # Relu関数 これがあるとないとではゴール到達率が違う defalt:n=0
        "=============================================="
        print(f"出力result [n={self.n} : {abs(neu_fire)}]")
        if neu_fire > 0:
            print("🔥発火🔥")
            self.save_s.append(round(ΔS-neu_fire, 2))
            ΔS = neu_fire
        else:
            print("💧発火しない💧")
            self.save_s.append(ΔS)
            ΔS = 0
        self.data_node.append(abs(neu_fire))
        self.XnWn_list.append(XnWn)
        print("[result] : ", self.data_node)
        print("[入力, 出力] : ", self.XnWn_list)

        return ΔS

    def hierarchical_model_X(self): # 良い状態ではない時に「戻るタイミングは半信半疑」とした時のストレス値の蓄積の仕方

        self.End_of_O = True # ○の連続が途切れたのでTrue

        self.M += 1
        # self.mmm+=1
        print("===== 🌟🌟🌟🌟🌟 =====")
        print("total : ", round(self.total_stress, 3))
        print("Save ΔS-Neuron : ", self.save_s)
        print("Save's Σ : ", self.Σ)
        "----- parameter -----" # Add self.Σ
        self.Σ = 1 # ×の時に蓄積する量は1.0とした
        self.n2 = copy.copy(self.n)
        "----- parameter -----"
        print("Save's Σ : ", self.Σ)
        print("[M, n2] : ", self.M, self.n2)
        print("[befor] total : ", round(self.total_stress, 3))
        print("m/m+n=", self.M/(self.M+self.n2))
        self.total_stress += self.Σ *1.0* (self.M/(self.M+self.n2)) # n=5,0.2 # ここ main # 階層化 ver.
        "階層化なしver."
        # self.total_stress += self.Σ # row
        print("[after] total : ", round(self.total_stress, 3))
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        "基準距離を可視化に反映させないver.はコメントアウト"
        # self.total_stress -= self.test_s # ×分は蓄積したので、基準距離分は一旦リセット
        "基準距離を可視化に反映させないver.はコメントアウト"

        print("[-基準距離] total : ", round(self.total_stress, 3))
        self.test_s = 0
        print("===== 🌟🌟🌟🌟🌟 =====")

        return True

    def match(self, Node, Arc):
        # pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        self.index = Node.index(self.NODELIST[self.state.row][self.state.column]) # これがselfではなかったので更新されなかった

        # print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
        print("Pre_Arc (事前のArc) : {}".format(Arc[self.index]))
        print("Act_Arc (実際のArc) : {}".format(self.test_s))
        # self.SAVE_ARC.append(self.test_s)
        print(f"Total Stress:{self.total_stress}")

        "========================================================================================================"
        "-- min-cost-cal-edit --"
        self.new = self.NODELIST[self.state.row][self.state.column]
        "-- min-cost-cal-edit --"
        LastNode = self.old # self.Node_l.index(self.old)
        NextNode = self.new # self.Node_l.index(self.new)
        self.old = self.new
        if not self.NODELIST[self.state.row][self.state.column] == "s":
            Act_Arc_data = self.move_step
        else:
            Act_Arc_data = 0
        cost_row = LastNode
        cost_column = NextNode

        if self.l.loc[cost_row, cost_column] == 0 or Act_Arc_data < self.l.loc[cost_row, cost_column]:
            self.l.loc[cost_row, cost_column] = Act_Arc_data

        Landmark = self.NODELIST[self.state.row][self.state.column]
        print(f"Landmark : {Landmark}")
        print(self.test_bp_st_pre[f"{Landmark}"])
        print("nan!!!!!")
        self.test_bp_st_pre[f"{Landmark}"] = self.state
        print("-----=========================================================================================\n")
        print(f"move step : {self.move_step}")
        print("  0,1,2,3,4,5,6,7,8,X")
        print(self.l)
        # print(f" X : {shortest_path(np.array(self.l), indices=X, directed=False)}")
        print(f"{shortest_path(np.array(self.l), directed=False)}")
        print("-----=========================================================================================\n")
        "-- min-cost-cal-edit --"
        print("-----=========================================================================================\n")
        print(f"test_bp_st: \n{self.test_bp_st_pre}")
        # # self.test_bp_st_pre.dropna(inplace=True)
        # print(self.test_bp_st_pre)
        # print("-----")
        # # self.test_bp_st_pre.drop(index=["x"], inplace=True)
        # print(self.test_bp_st_pre)
        "========================================================================================================"

        try:
            kizyun_d = self.move_step/float(Arc[self.index])
        except:
            kizyun_d = 0
        print("move step = ", self.move_step)
        print(Arc)
        print(self.index)
        print("事前 = ", float(Arc[self.index]))
        print("基準d = ", kizyun_d) # これを基準ストレスにする
        if kizyun_d != 0:
            "-- これがいずれのΔSnodeの式 今はArc に対するΔSのみ --"
            if kizyun_d > 2:
                kizyun_d = 0.0
            kizyun_d = round(abs(1.0-kizyun_d), 3)
        else:
            kizyun_d = 0.5 # 0.0 start 地点
        print("ΔS_Arc【基準ストレス】 : {}".format(kizyun_d))

        if not self.NODELIST[self.state.row][self.state.column] == "s":
            # self.SAVE_ARC.append(round(self.test_s*float(Arc[self.index]), 2))
            self.SAVE_ARC.append(round(self.move_step, 2))
        self.move_step = 0

        print("⚠️ 実際のアークの配列 : {}".format(self.SAVE_ARC))
        print("Arc[self.index]:{}".format(float(Arc[self.index])))
        # print("----\n今の permission : {} 以内に発見\n----".format(PERMISSION[self.index][0]))

        "====================================== 追加部分 =========================================="
        ΔS = 0.3 # ここも基準距離に対するストレスにする
        self.save_s_all.append(ΔS)

        ΔS = self.hierarchical_model_O(ΔS) # 関数 これがないとゴール到達率が下がる
        
        print("==========================================")
        print("SUM : ", self.total_stress)
        print("ΔS Arc : ", kizyun_d)
        print("ΔS : ", ΔS)
        print("Save ΔS-Neuron : ", self.save_s)
        print("Save's Σ : ", round(sum(self.save_s), 2))
        self.Σ = round(sum(self.save_s), 2)
        print("Save ΔS : ", self.save_s_all)
        print("Save's All Σ : ", round(sum(self.save_s_all), 2))
        print("==========================================")

        self.n_m[self.state.row][self.state.column] = (self.n, self.M) # 連続数(n, m)の追加
        pprint.pprint(self.n_m)
        self.phi = [self.n, self.M]
        print("👍 (adv++) phi = ", self.phi)
        
        "====================================== 追加部分 =========================================="
        print("ΔS_Arc arc stress【基準ストレス】 : {}".format(kizyun_d))  #このままだとArcが大きくなるとストレス値も大きくなってしまい、ストレス値の重みが変わってしまうので、基準[1]にする

        "===================================================================="
        "Nodeに対するストレスの保存"
        "== 基準距離でノードに対するストレス + stressの小ささで戻るノードを決める場合 =="
        self.Observation[self.state.row][self.state.column] = round(abs(kizyun_d), 3)
        "全部コメントアウトの時はsettingのobservationの数値をそのまま使う"
        "===================================================================="
        pprint.pprint(self.Observation)
        try:
            self.OBS.append(self.Observation[self.state.row][self.state.column])
        except:
            self.OBS = self.OBS.tolist()
            self.OBS.append(self.Observation[self.state.row][self.state.column])
        print("OBS : {}".format(self.OBS))

        self.Add_Advance = True
        self.BPLIST.append(self.state)

        # 一個前が1ならpopで削除
        print("📂 Storage {}".format(self.BPLIST))
        print("Storage append : {}".format(self.Storage))

        "BPLISTを保存"
        for bp, stress in zip(self.BPLIST, self.OBS):
            if bp not in self.Storage:
                self.Storage.append(bp)
                self.Storage_Stress.append(stress)
        print("Storage append : {}".format(self.Storage))
        print("Storage Stress append : {}".format(self.Storage_Stress))
        print("Storage Arc : {}".format(self.Storage_Arc))

        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        self.test_s = 0
        
        "基準距離を可視化に反映させないver.はコメントアウト"
        # self.total_stress = 0
        # self.total_stress += arc_s
        "基準距離を可視化に反映させないver.はコメントアウト +代わりに以下"
        if not self.NODELIST[self.state.row][self.state.column] == "s": # これはスタート地点にノードを設定している場合、初期位置ではストレスを蓄積させないため
            self.total_stress += ΔS # 基準距離を可視化させないver.
        self.SIGMA_LIST.append(self.total_stress)
        print("SIGMA : {}".format(self.SIGMA_LIST))
        print("Total Stress (減少後) : {}".format(self.total_stress))

    def nomatch(self):

        judge_node__x = False

        if self.grid[self.state.row][self.state.column] == 5:
            print("\n\n\n交差点! 🚥　🚙　✖️")
            if self.state not in self.CrossRoad:
                print("\n\n\n未探索の交差点! 🚥　🚙　✖️")
                self.CrossRoad.append(self.state)
            print("CrossRoad : {}\n\n\n".format(self.CrossRoad))
        print("事前情報にないNode!!!!!!!!!!!!")
        if self.NODELIST[self.state.row][self.state.column] == "x":
            true_or_false = self.hierarchical_model_X()

            self.n_m[self.state.row][self.state.column] = (self.n, self.M) # 連続数(n, m)の追加
            pprint.pprint(self.n_m)
            self.phi = [self.n, self.M]
            print("👍 (adv × ) phi = ", self.phi)

            
            "===== パラメータ =====" # row ver. はコメントアウト
            if self.M/(self.M+self.n) >= 0.5: # 0.3: # 階層化 ver.
                self.TRIGAR = True
                self.COUNT += 1
                self.BPLIST.append(self.state)
                self.Add_Advance = True

                judge_node__x = True
        
        return judge_node__x

    def threshold(self):
        
        self.TRIGAR = True
        self.COUNT += 1
        self.BPLIST.append(self.state) # Arcを計算する為に、最初だけ必要
        self.Add_Advance = True
        "============================================== Visualization ver. との違い =============================================="
        print(f"🤖 State:{self.state}")
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        print(f"Total Stress:{self.total_stress}")

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×
        
        self.SAVE_ARC.append(round(self.move_step, 2))

        "----- min cost cal -----"
        print("-----=========================================================================================\n")
        print(f"move step : {self.move_step}")
        self.new = "x"
        LastNode = self.Node_l.index(self.old)
        X = self.Node_l.index(self.new)

        Act_Arc_data = self.move_step
        cost_row = self.old # LastNode
        cost_column = self.new # X # NextNode -> "x"
        self.l.loc[cost_row, cost_column] = Act_Arc_data # 戻る場所からNodeまでの距離を一時的に最小値とか関係なく格納する

        print(self.l)
        print(f"{shortest_path(np.array(self.l), directed=False)}")
        print("----- 始点 = x の場合 -----")
        print("Node : 0,  1,  2,  3,  4,  5,  6,  7,  8,  X")
        print(f" X : {shortest_path(np.array(self.l), indices=X, directed=False)}")


        self.move_cost_result_X = shortest_path(np.array(self.l), indices=X, directed=False)
        self.move_cost_result = self.l
        print("-----=========================================================================================\n")

        self.l.loc[cost_row, cost_column] = 0 # これが重要 戻り始める場所は毎回変わるのでリセットする

        "----- min cost cal -----"

    def trigar(self):

        self.env.mark(self.state, self.TRIGAR)
        print("終了します")
        self.BPLIST.append(self.state) # Arcを計算する為に、最初だけ必要
        self.Add_Advance = True
        
        self.SAVE_ARC.append(round(self.move_step, 2))

        print("-----=========================================================================================\n")
        print(f"move step : {self.move_step}")
        self.new = "x"
        LastNode = self.Node_l.index(self.old)
        X = self.Node_l.index(self.new)
        Act_Arc_data = self.move_step
        cost_row = self.old # LastNode
        cost_column = self.new # X # NextNode -> "x"
        self.l.loc[cost_row, cost_column] = Act_Arc_data # 戻る場所からNodeまでの距離を一時的に最小値とか関係なく格納する
        print(self.l)
        print(f"{shortest_path(np.array(self.l), directed=False)}")
        print("----- 始点 = x の場合 -----")
        print("Node : 0,  1,  2,  3,  4,  5,  6,  7,  8,  X")
        print(f" X : {shortest_path(np.array(self.l), indices=X, directed=False)}")
        self.move_cost_result_X = shortest_path(np.array(self.l), indices=X, directed=False)
        self.move_cost_result = self.l
        print("-----=========================================================================================\n")

        self.l.loc[cost_row, cost_column] = 0 # これが重要 戻り始める場所は毎回変わるのでリセットする

    def Advance(self, STATE_HISTORY, state, TRIGAR, OBS, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, move_step, old_from_exp, move_cost_result, test_bp_st, Backed_just_before, phi, standard_list, rate_list, test_s):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        self.grid = grid
        self.total_stress = total_stress # 今はストレス値は共有していないのでいらない
        self.OBS = OBS
        self.action = random.choice(self.env.actions) # コメントアウト 何も処理されない時はこれが prev action に入る
        self.Add_Advance = False
        self.Backed_just_before = Backed_just_before
        self.phi = phi
        GOAL = False
        self.CrossRoad = CrossRoad
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.stress = 0
        self.index = Node.index("s")
        pprint.pprint(pre)
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        self.test_s = test_s
        self.move_step = move_step
        self.old = old_from_exp

        print(f"========== test self.l:\n{self.l}")
        ΔS = 0

        if self.Backed_just_before: # 直前で戻っていた場合 これはbp.pyにてself.Backed_just_before = Trueを追加する
            pprint.pprint(self.n_m)
            print("👍 (adv) phi = ", self.phi)
            self.n = phi[0]
            self.M = phi[1]
        else: # 初期値
            self.n = phi[0] # 1
            self.M = phi[1] # 1
            # self.nnn=1
            # self.mmm=1

        while not self.done:
        
            print("\n-----{}Steps-----".format(self.COUNT+1))
            
            self.move_step += 1

            self.map_unexp_area = self.env.map_unexp_area(self.state)
            if self.map_unexp_area or self.FIRST:
                    self.FIRST = False
                    print("un explore area ! 🤖 ❓❓")
                    if self.test_s + self.stress >= 0:

                        # 蓄積量(傾き)
                        ex = (self.n/(self.n+self.M))
                        ex = -2*ex+2
                        try:
                            self.test_s += round(self.stress/float(Arc[self.index-1]), 3) *ex
                            "基準距離を可視化に反映させないver.はコメントアウト"
                        except:
                            self.test_s += 0
                            "基準距離を可視化に反映させないver.はコメントアウト"

                        print("Arc to the next node : {}".format(Arc[self.index-1]))

                    if self.NODELIST[self.state.row][self.state.column] in pre:
                        
                        print("🪧 NODE : ⭕️")
                        print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))

                        if self.NODELIST[self.state.row][self.state.column] == "g":
                            print("🤖 GOALに到達しました。")
                            GOAL = True
                            self.STATE_HISTORY.append(self.state)
                            self.TOTAL_STRESS_LIST.append(self.total_stress)

                            "基準距離, 割合の可視化"
                            self.standard_list.append(self.test_s)
                            # self.rate_list.append(self.n/(self.M+self.n))    # ○
                            self.rate_list.append(self.M/(self.M+self.n))      # ×
                            
                            break

                        self.match(Node, Arc) 

                    else:

                        print("🪧 NODE : ❌")
                        print("no match!")

                        judge_node__x = self.nomatch()

                        if judge_node__x:

                            print("=================")
                            print("FULL ! MAX! 🔙⛔️")
                            print("=================")

                            self.threshold()
                            
                            break

                    if self.test_s >= 2.0: # 基準距離で判断 階層化ver.
                    # if self.test_s >= 2.0 or self.total_stress >= 2.0: # row ver.
                        
                        print("基準距離 = ", self.test_s)
                        print(f"Total Stress:{self.total_stress}")
                        print("=================")
                        print("FULL ! MAX! 🔙⛔️")
                        print("=================")

                        self.threshold()
                        

                        break
            else:
                print("================\n🤖 何も処理しませんでした__2\n================")
                print("マーキング = 1 の探索済みエリア")
                
            print(f"🤖 State:{self.state}")
            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)
            print(f"Total Stress:{self.total_stress}")
            print("基準距離 = ", self.test_s)

            "基準距離, 割合の可視化"
            self.standard_list.append(self.test_s)
            # self.rate_list.append(self.n/(self.M+self.n)) # ○
            self.rate_list.append(self.M/(self.M+self.n))   # ×

            self.action, self.Reverse, self.TRIGAR = self.agent.policy_advance(self.state, self.TRIGAR, self.action)
            if self.TRIGAR:

                print("Trigar")
                print("ストレスが溜まり切る前にこれ以上進めい")

                self.trigar()
                
                
                break


            self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
            self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
            self.state = self.next_state

            print("COUNT : {}".format(self.COUNT))
            if self.COUNT > 150:
                break
            self.COUNT += 1

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.OBS, self.BPLIST, self.action, self.Add_Advance, GOAL, self.SAVE_ARC, self.CrossRoad, self.Storage, self.Storage_Stress, self.TOTAL_STRESS_LIST, self.move_cost_result, self.test_bp_st_pre, self.move_cost_result_X, self.standard_list, self.rate_list