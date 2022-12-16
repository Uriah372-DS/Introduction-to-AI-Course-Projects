
def twoSum(nums: list[int], target: int) -> list[int]:
    required = {}
    n = len(nums)
    for i in range(n):
        if target - nums[i] in required:
            return [required[target - nums[i]], i]
        else:
            required[nums[i]] = i
#  time complexity - O(n)


def question2(prices: list[int]) -> int:
    max_profit = 0
    n = len(prices)
    for i in range(n):
        for j in range(i + 1, n):
            profit = prices[j] - prices[i]
            if profit > max_profit:
                max_profit = profit
    return max_profit


class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

    def __str__(self):
        return str(self.value)


def read_file(file_path: str) -> Node:
    with open(file_path, 'r') as f:
        dummy = Node(value=0)
        x = dummy
        for line in f:
            for value in str(line).split(sep=','):
                x.next = Node(value=int(value))
                x = x.next
    head = dummy.next
    del dummy
    return head


def get_length(head: Node) -> int:
    length = 0
    x = head
    while x is not None:
        x = x.next
        length += 1
    return length


def sort_in_place(head: Node) -> Node:
    x = head
    while x is not None:
        y = x.next
        min_val = x.value
        min_node = x
        while y is not None:
            if y.value < min_val:
                min_val = y.value
                min_node = y
            y = y.next
        temp = x.value
        x.value = min_val
        min_node.value = temp
        x = x.next
    return head
#  time complexity - O(n^2)  (min-sort)
#  space complexity - O(1)


if __name__ == '__main__':
    print(twoSum(nums=[2, 8, 11, 15], target=10))
    print(question2(prices=[7, 1, 5, 3, 6, 4]))
    test = read_file(file_path="test")
    print(get_length(test))
    sorted_test = sort_in_place(test)
    sorted_test_list = []
    while sorted_test is not None:
        sorted_test_list.append(sorted_test.value)
        sorted_test = sorted_test.next
    print(sorted_test_list)

