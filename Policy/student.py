from Policy.policy import Policy
import numpy as np


# 1: Column Generation, 2: Branch and Bound, 3: First Fit Decreasing
class Policy2352234(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id 

            # # Parameters for branch and bound
            # self.best_actions = []
            # self.best_filled = float('inf')
            # self.root = None
            # self.action_count = 0  # Đếm số lượng hành động đã thực hiện trong best_actions

        self.last_prod_w = 0
        self.last_prod_h = 0
        self.last_stock_idx = 0
        self.last_pos_x = 0
        self.last_pos_y = 0

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing_action(observation, info)
        elif self.policy_id == 2:
            # First Fit Decreasing
            return self.best_fit_decreasing_action(observation, info)

    ############################################################################################################
    # First Fit Decreasing    
    def first_fit_decreasing_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        sorted_stock_incidies = self.sort_stock(observation)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                return self.get_action_for_product(prod, observation, sorted_stock_incidies)

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def area(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)

        return stock_w * stock_h

    def sort_stock(self, observation):
        sorted_stock_incidies = [i for i in range(len(observation["stocks"]))]
        sorted_stock_incidies = sorted(sorted_stock_incidies, key=lambda x: self.area(observation["stocks"][x]), reverse=True)
        return sorted_stock_incidies
    
    def get_action_for_product(self, prod, observation, stock_incidies):
        prod_size = prod["size"]
        begin_stock_idx_ = 0

        # for time improvement
        if ((self.last_prod_w == prod_size[0] 
                and self.last_prod_h == prod_size[1]) 
            or (self.last_prod_w == prod_size[1] 
                and self.last_prod_h == prod_size[0])):
            begin_stock_idx_ = self.last_stock_idx

        for i in range(begin_stock_idx_, len(stock_incidies)):
            stock = observation["stocks"][stock_incidies[i]]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            if stock_w >= prod_w and stock_h >= prod_h:
                # pos_x, pos_y = None, None
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            self.last_prod_w = prod_size[0]
                            self.last_prod_h = prod_size[1]
                            self.last_stock_idx = i
                            self.last_pos_x = x
                            self.last_pos_y = y

                            return {"stock_idx": stock_incidies[i], "size": prod_size, "position": (x, y)}

            if stock_w >= prod_h and stock_h >= prod_w:
                # pos_x, pos_y = None, None
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                            self.last_prod_w = prod_size[1]
                            self.last_prod_h = prod_size[0]
                            self.last_stock_idx = i
                            self.last_pos_x = x
                            self.last_pos_y = y
                            return {"stock_idx": stock_incidies[i], "size": prod_size[::-1], "position": (x, y)}
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    ############################################################################################################
    # Best Fit Decreasing
    def best_fit_decreasing_action(self, observation, info):
        list_prods = [prod for prod in observation["products"] if prod["quantity"] > 0]
        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        stock_incidies = [i for i in range(len(observation["stocks"]))]
        stock_incidies = self.sort_stock(observation) # sort by number of empty cells

        for prod in sorted_prods:
            prod_size = prod["size"]

            begin_stock_idx = 0
            if (prod_size[0] == self.last_prod_w and prod_size[1] == self.last_prod_h) or (prod_size[0] == self.last_prod_h and prod_size[1] == self.last_prod_w):
                begin_stock_idx = self.last_stock_idx
            
            
            # fill height first
            if prod_size[0] > prod_size[1]:
                prod_size = prod_size[::-1]

            for i in range(begin_stock_idx, len(stock_incidies)):
                stock = observation["stocks"][stock_incidies[i]]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                self.last_prod_w = prod_size[0]
                                self.last_prod_h = prod_size[1]
                                self.last_stock_idx = i
                                return {"stock_idx": stock_incidies[i], "size": prod_size, "position": (x, y)}
                            
                if stock_w >= prod_h and stock_h >= prod_w:
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):        
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                self.last_prod_w = prod_size[1]
                                self.last_prod_h = prod_size[0]
                                self.last_stock_idx = i
                                return {"stock_idx": stock_incidies[i], "size": prod_size[::-1], "position": (x, y)}
                            
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}