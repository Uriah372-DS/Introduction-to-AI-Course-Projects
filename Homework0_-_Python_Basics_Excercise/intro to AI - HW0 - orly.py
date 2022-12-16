# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from itertools import zip_longest, combinations


class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node

    def __str__(self):
        return str(self.value)


# Q1
# naive implementation - O(n^2) time complexity
def naive_twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]


# Q1
# O(n) time complexity implementation
def twoSum(nums, target):
    # we can remove duplicates from the list as they don't matter for
    # this question
    unique_nums = set(nums)
    for n in combinations(unique_nums, 2):
        if n[0] + n[1] == target:
            return [n[0], n[1]]


# Q2
def maximize_profit(prices):
    results = []
    start = 0
    for i, (a, b) in enumerate(zip_longest(prices, prices[1:])):
        if b is None or b <= a:
            results.append(prices[start:i + 1])
            start = i + 1
    profits = []
    for i in results:
        profits.append(max(i) - min(i))
    return max(profits)


# Q3.1
def read_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read().split(';')
    cur = head = Node(0)
    for i in file_content:
        cur.next = Node(int(i))
        cur = cur.next
    return head.next


# Q3.2
def get_length(head):
    count = 0
    curr_node = head
    while curr_node is not None:
        curr_node = curr_node.next
        count += 1
    return count


# Q3.3
def merge(head1, head2):
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    if head1.value <= head2.value:
        head = head1
        head1 = head1.next
    else:
        head = head2
        head2 = head2.next
    curr = head
    while head1 and head2:
        if head1.value <= head2.value:
            curr.next = head1
            head1 = head1.next
        else:
            curr.next = head2
            head2 = head2.next
        curr = curr.next
    if head1 is None:
        curr.next = head2
    else:
        curr.next = head1
    return head


def find_midpoint(head):
    mid = head
    for i in range(1, (get_length(head)) // 2):
        mid = mid.next
    return mid


def sort_in_place(head):
    # Sort a given linked list using Merge Sort.
    if head is None:
        return None
    if head.next is None:
        return head
    # We have at least 2 nodes. Split the list into two parts equally
    mid = find_midpoint(head)
    head2 = mid.next
    mid.next = None
    # MergeSort the two lists
    head = sort_in_place(head)
    head2 = sort_in_place(head2)
    head = merge(head, head2)
    return head
