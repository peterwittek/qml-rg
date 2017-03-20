+import random
+import numpy as np
+
+
+class Game(object):
+
+    def __init__(self):
+        self.state = [0, 0, 0]
+        self.agent_position = 1
+        self.state[random.randint(0, 1)*2] = 1
+        self.status = "running"
+
+    def moveto(self, position):
+        self.agent_position += position
+        if self.agent_position < 0:
+            self.agent_position = 0
+        elif self.agent_position > 2:
+            self.agent_position = 2
+        if self.status == "running" and self.state[self.agent_position] == 1:
+            self.status = "gameover"
+            return 1
+        else:
+            return 0
+
+
+class BabyAgent(object):
+
+    def __init__(self, game):
+        self.game = game
+        self.number_of_steps = 0
+        self.weights = np.array([1, 1])
+        self.step = 0
+
+    def next_move(self):
+        self.number_of_steps += 1
+        if game.agent_position != 1:
+            self.weights[(self.decision + 1) % 2] += 1
+        self.decision = np.random.choice(2, 1,
+                                         p=self.weights/sum(self.weights))
+        if self.decision == 0:
+            self.step = -1
+        else:
+            self.step = 1
+        return self.step
+
+game = Game()
+print(game.state)
+agent = BabyAgent(game)
+while game.status == "running":
+    agent_move = agent.next_move()
+    game.moveto(agent_move)
+    print(agent.weights)
+    print(agent.decision)
+
+print(agent.number_of_steps)
