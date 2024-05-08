import networkx as nx
import matplotlib.pyplot as plt

class RBNode:
    def __init__(self, key, color='RED'):
        self.key = key
        self.parent = None
        self.left = None
        self.right = None
        self.color = color

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(None, color='BLACK')  # Sentinel node
        self.root = self.NIL

    def insert(self, key):
        new_node = RBNode(key)
        parent = None
        current = self.root
        while current != self.NIL:
            parent = current
            if key < current.key:
                current = current.left
            else:
                current = current.right
        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.left = self.NIL
        new_node.right = self.NIL
        new_node.color = 'RED'
        self._insert_fixup(new_node)

    def _insert_fixup(self, node):
        while node != self.root and node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._left_rotate(node.parent.parent)
        self.root.color = 'BLACK'

    def delete(self, key):
        node = self._search(key)
        if node is None:
            return
        self._delete(node)

    def _delete(self, node):
        if node.left == self.NIL or node.right == self.NIL:
            y = node
        else:
            y = self._tree_minimum(node.right)
        if y.left != self.NIL:
            x = y.left
        else:
            x = y.right
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        if y != node:
            node.key = y.key
        if y.color == 'BLACK':
            self._delete_fixup(x)

    def _delete_fixup(self, node):
        while node != self.root and node.color == 'BLACK':
            if node == node.parent.left:
                sibling = node.parent.right
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self._left_rotate(node.parent)
                    sibling = node.parent.right
                if sibling.left.color == 'BLACK' and sibling.right.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    if sibling.right.color == 'BLACK':
                        sibling.left.color = 'BLACK'
                        sibling.color = 'RED'
                        self._right_rotate(sibling)
                        sibling = node.parent.right
                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.right.color = 'BLACK'
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self._right_rotate(node.parent)
                    sibling = node.parent.left
                if sibling.right.color == 'BLACK' and sibling.left.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    if sibling.left.color == 'BLACK':
                        sibling.right.color = 'BLACK'
                        sibling.color = 'RED'
                        self._left_rotate(sibling)
                        sibling = node.parent.left
                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.left.color = 'BLACK'
                    self._right_rotate(node.parent)
                    node = self.root
        node.color = 'BLACK'

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _search(self, key):
        current = self.root
        while current != self.NIL and key != current.key:
            if key < current.key:
                current = current.left
            else:
                current = current.right
        return current

    def _tree_minimum(self, node):
        while node.left != self.NIL:
            node = node.left
        return node

    def visualize(self):
        G = nx.DiGraph()
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node != self.NIL:
                if node.left != self.NIL:
                    G.add_edge(node.key, node.left.key)
                    queue.append(node.left)
                else:
                    G.add_node(node.left)
                if node.right != self.NIL:
                    G.add_edge(node.key, node.right.key)
                    queue.append(node.right)
                else:
                    G.add_node(node.right)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue')
        plt.show()

# Пример использования:
if __name__ == "__main__":
    rb_tree = RedBlackTree()
    rb_tree.insert(15)
    rb_tree.insert(24)
    rb_tree.insert(10)
    rb_tree.insert(7)
    rb_tree.visualize()

    rb_tree.delete(15)

    rb_tree.visualize()

