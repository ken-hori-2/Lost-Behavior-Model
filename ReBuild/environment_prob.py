from enum import Enum
from pprint import pprint
import pprint
from random import random
import random
import numpy as np
from reference_match_rate_Robosin import Property


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "[{}, {}]".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column
        
class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2

class Environment():

    def __init__(self, *arg, move_prob = 1.0): # 0.8):
        
        self.agent_state = State()
        self.reset()
        self.grid = arg[0]
        self.map = arg[1]
        self.NODELIST = arg[2]
        self.default_stress = 1
        self.refer = Property()
        self.move_prob = move_prob

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    def reset(self):
        self.agent_state = State(22, 8)
        
        return self.agent_state

    def can_action_at(self, state):

        if self.grid[state.row][state.column] == 5:
            print("===== 交差点 =====")
            return True
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action, TRIGAR):

        if not self.can_action_at(state):
            
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
            
        if not (0 <= next_state.column < self.column_length):
            next_state = state
            

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        # stress, done = self.stress_func(next_state, TRIGAR) # 上だと沼る
        # self.mark(next_state, TRIGAR)

        return next_state # , stress, done

    def stress_func(self, state, TRIGAR):

        
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
       
        done = False

        # Check an attribute of next state.
        attribute = self.NODELIST[state.row][state.column]

        if TRIGAR:
            # stress = -self.default_stress
            stress = 0
        else:
            # if attribute in pre:
            #     # Get reward! and the game ends.
            #     print("###########")
            #     stress = 0 # -1                              # ここが reward = None の原因 or grid の 1->0 で解決
            # else:
                stress = self.default_stress

        return stress, done

    def transit_func(self, state, action, TRIGAR):
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        opposite_direction = Action(action.value * -1) # 進もうとしている方向と逆方向を格納

        print("opposite direction : {}".format(opposite_direction))

        for a in self.actions: # 進もうとしている左右方向に確率を振る
            prob = 0
            if a == action:
                # prob = 1 
                prob = self.move_prob
            # elif a != opposite_direction: # 進もうとしている方向と　逆以外　なら確率を半分ずつ与える
            #     # prob = 0 
            #     prob = (1 - self.move_prob) / 2
            # elif a == opposite_direction: # 進もうとしている方向と　逆　なら確率を半分ずつ与える
                # prob = 0 
            else:
                prob = (1 - self.move_prob) / 3

            next_state = self._move(state, a, TRIGAR)

            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

            if a == action:
                self.next_state_plan = next_state
                
        pprint.pprint(self.map)

        return transition_probs
        
    def step(self, state, action, TRIGAR):
        self.agent_state = state
        next_state, stress, done = self.transit(self.agent_state, action, TRIGAR)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, stress, done

    def transit(self, state, action, TRIGAR):
        transition_probs = self.transit_func(state, action, TRIGAR)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)

        if next_state == self.next_state_plan:
            print("\n=====\n想定方向に行動しました!!!!!!!!!!!!!!!!\n=====\n")
            print("next state plan : {}".format(self.next_state_plan))
            print("next state : {}".format(next_state))
        else:
            print("\n=====\n想定方向に行動出来ませんでした!!!!!!!!!!!!!!!!\n=====\n")
            print("next state plan : {}".format(self.next_state_plan))
            print("next state : {}".format(next_state))

            # if opposite_direction == action:
            oppsite_next_state = self.expected_next_state(state, action)
            if oppsite_next_state == next_state:
                print("insite next state : {}".format(oppsite_next_state))
                print("oppsite next state : {}".format(oppsite_next_state))
                print("\n=====\nかつ、想定と逆方向に行動しました!!!!!!!!!!!!!!!!\n=====\n")
                self.mark_miss(state)

        stress, done = self.stress_func(next_state, TRIGAR)
        return next_state, stress, done



    def mark(self, state, TRIGAR):

        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()

        attribute = self.NODELIST[state.row][state.column]

        # if attribute in pre:
        self.map[state.row][state.column] = self.marking_param # 1
  
    def mark_all(self, state):

        self.map[state.row][state.column] = 2

        # pprint.pprint(self.map)

    def mark_reverse(self, state):

        self.map[state.row][state.column] = 1

        # pprint.pprint(self.map)

    def mark_miss(self, state):
        
        if self.map[state.row][state.column] > 0:
            self.map[state.row][state.column] -= 1 # = 0
        # if self.grid[state.row][state.column] == 5: # これだけだと交差点前に間違えた時に交差点に進めなくなる
        #     print("===== 交差点 =====")
        #     self.map[state.row][state.column] = 1

    def expected_move(self, state, action, TRIGAR, All, marking_param):
        
        next_state = state.clone()
        test = True
        self.marking_param = marking_param
        self.mark(state, TRIGAR)
        print("\nMARKING : {}".format(marking_param))
        # pprint.pprint(self.map)

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        
        if not (0 <= next_state.row < self.row_length):
            next_state = state
            test = False
            
        if not (0 <= next_state.column < self.column_length):
            next_state = state
            test = False

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state
            test = False
        
        if self.map[next_state.row][next_state.column] >= marking_param: # == 1:
            next_state = state
            test = False

        return test, action

    # move_returnと同じ
    def expected_not_move(self, state, action, TRIGAR, REVERSE):

        next_state = state.clone()
        test = False
        self.mark_all(state)

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1
        
        if not (0 <= next_state.row < self.row_length):
            next_state = state
            test = False
            
        if not (0 <= next_state.column < self.column_length):
            next_state = state
            test = False

        # Check whether the agent bumped a block cell.
        # if self.grid[next_state.row][next_state.column] == 9:
        #     next_state = state
        #     test = False
        
        # if self.map[next_state.row][next_state.column] == 1:
        #     next_state = state
        #     test = True
            
        if self.map[next_state.row][next_state.column] == 2:
            next_state = state
            # pprint.pprint(self.map)
            test = True

        return test, action

    def map_unexp_area(self, state):
        if self.map[state.row][state.column] == 0:
            return True
        else:
            return False






    def expected_next_state(self, state, action):
        
        next_state = state.clone()

        oppsite_next_state = state.clone()

        test = True

        opposite_direction = Action(action.value * -1)

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        if opposite_direction == Action.UP:
            oppsite_next_state.row -= 1
        elif opposite_direction == Action.DOWN:
            oppsite_next_state.row += 1
        elif opposite_direction == Action.LEFT:
            oppsite_next_state.column -= 1
        elif opposite_direction == Action.RIGHT:
            oppsite_next_state.column += 1

        return oppsite_next_state