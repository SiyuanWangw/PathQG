class Graph(object):
    def __init__(self, sentence, answers, questions):
        self.sentence = sentence
        self.answers = answers
        self.questions = questions
        self.node_list = []
        self.edge_list = []
        self.node_text_list = []
        self.edge_text_list = []

    def extend_graph_node(self, nodetext, nodeattr, nodespan_bounds, nodetype):
        new_node = Node(nodetext, nodeattr, nodespan_bounds, nodetype)
        self.node_list.append(new_node)
        self.node_text_list.append(nodetext)

    def extend_graph_edge(self, edgetext, edgetype, subjectNode, objectNode):
        new_edge = Edge(edgetext, edgetype, subjectNode, objectNode)
        self.edge_list.append(new_edge)
        self.edge_text_list.append(edgetext)
        subjectNode.add_succ(objectNode, new_edge)
        objectNode.add_pre(subjectNode, new_edge)

    def get_node(self, node_text):
        if node_text in self.node_text_list:
            node_index = self.node_text_list.index(node_text)
            return self.node_list[node_index]

    def get_edge(self, sub_node, ob_node):
        for each_edge in self.edge_list:
            if each_edge.subject.nodetext == sub_node.nodetext and each_edge.object.nodetext == ob_node.nodetext:
                return each_edge

    def get_edge_by_index(self, i, j):
        cur_node = self.node_list[i]
        if self.node_list[j] in cur_node.succNodes:
            find_index = cur_node.succNodes.index(self.node_list[j])
            return cur_node.succEdges[find_index]
        else:
            return


class Node(object):
    def __init__(self, nodetext, nodeattr, nodespan_bounds, nodetype):
        self.nodetext = nodetext
        self.nodeattr = nodeattr
        self.nodespan_bounds = nodespan_bounds
        self.nodetype = nodetype
        self.preNodes = []
        self.preEdges = []
        self.succNodes = []
        self.succEdges = []
        self.parent = None

    def add_succ(self, succ_node, succ_edge):
        self.succNodes.append(succ_node)
        self.succEdges.append(succ_edge)

    def add_pre(self, pre_node, pre_edge):
        self.preNodes.append(pre_node)
        self.preEdges.append(pre_edge)

    def add_parent(self, parent_node):
        self.parent = parent_node

    # def add_parent(self, parent_node):
    #     self.parent.append(parent_node)

    def set_type(self, nodetype):
        self.nodetype = nodetype


class Edge(object):
    def __init__(self, edgetext, edgetype, subjectNode, objectNode):
        #############################################
        # edgetext: text between two nodes
        # edgetype: preposition and verb
        #############################################
        self.edgetext = edgetext
        self.edgetype = edgetype
        self.subject = subjectNode
        self.object = objectNode