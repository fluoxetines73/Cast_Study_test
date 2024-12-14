
from Policy.policy import GreedyPolicy, RandomPolicy
from Policy.student import Policy2352234
import time
import numpy as np

class Case_Study_Env():
    def __init__(self, stock_size=[60, 40], product_sizes=[[30, 24], [56, 13], [22, 14], [23, 9]], product_demands=[100, 122, 115, 156]):
        self.stock_size = stock_size
        self.product_sizes = product_sizes
        self.product_demands = product_demands
        self.observation = self.init_observation(stock_size, product_sizes, product_demands)
        self.info = self._get_info()

    def init_observation(self, stock_size, product_sizes, product_demands):
        stocks = np.full(stock_size, -1) # when start, there is one stock with all empty cells
        products = []
        for size, demand in zip(product_sizes, product_demands):
            products.append({"size": size, "quantity": demand})

        observation = {
            "stocks": [
                stocks
            ],
            "products": products
        }
        return observation
    
    def _step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        # Check if the product is in the product list
        product_idx = None
        for i, product in enumerate(self.observation["products"]):
            if np.array_equal(product["size"], size) or np.array_equal(
                product["size"], size[::-1]
            ):
                if product["quantity"] == 0:
                    continue

                product_idx = i  # Product index starts from 0
                break

        if product_idx is not None:
            # must make sure stock
            stock = self.observation["stocks"][stock_idx]
            # Check if the product fits in the stock
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))
            if (
                x >= 0
                and y >= 0
                and x + width <= stock_width
                and y + height <= stock_height
            ):
                # Check if the position is empty
                if np.all(stock[x : x + width, y : y + height] == -1):
                    # self.cutted_stocks[stock_idx] = 1
                    stock[x : x + width, y : y + height] = product_idx
                    self.observation["products"][product_idx]["quantity"] -= 1
                else:
                    print("Position is not empty")
                    pass
            else:
                print("Product does not fit in the stock")
                pass

        else:
            print("Product not found")
            pass

        # An episode is done iff the all product quantities are 0
        terminated = all([product["quantity"] == 0 for product in self.observation["products"]])
        # reward = 1 if terminated else 0  # Binary sparse rewards

        if np.any(self.observation["stocks"][-1] != -1):
            self.observation["stocks"].append(np.full(self.stock_size, -1))

        self.info = self._get_info()

        return self.observation, terminated, self.info
    
    def _get_info(self):
        cutted_stocks = [np.any(stock >= 0) for stock in self.observation["stocks"]]

        filled_ratio = np.sum(cutted_stocks).item()
        trim_loss = []

        for sid, stock in enumerate(self.observation["stocks"]):
            if cutted_stocks[sid] == 0:
                continue
            tl = (stock == -1).sum() / (stock != -2).sum()
            trim_loss.append(tl)

        trim_loss = np.mean(trim_loss).item() if trim_loss else 1

        return {"used_stock": filled_ratio, "trim_loss": trim_loss}
    
    def reset(self):
        self.observation = self.init_observation(self.stock_size, self.product_sizes, self.product_demands)
        self.info = self._get_info()
        return self.observation, self.info

if __name__ == "__main__":

    # stock_size = [60, 40]
    # products_size = [[30, 24], [56, 13], [22, 14], [23, 9]]
    # products_demand = [100, 122, 115, 156]

    stock_size = [88, 59]

    products_size = [[13, 9], [18, 14], [40, 28], [50, 13], [9, 5], [18, 14], [40, 28], [50, 13]]
    
    products_demand = [100, 122, 115, 156, 100, 122, 115, 156]

    env = Case_Study_Env(stock_size=stock_size, product_sizes=products_size, product_demands=products_demand)

    env.reset()
    print("====================================")
    ffd = Policy2352234(policy_id=1)

    start = time.time()
    while True:
        action = ffd.get_action(env.observation, env.info)
        obs, done, info = env._step(action)
        if done:
            end = time.time()
            break

    print("Time: ", end - start)

    print("====================================")
    print(info)
