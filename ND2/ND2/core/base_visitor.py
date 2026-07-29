from abc import ABC, abstractmethod
from .symbols import Symbol

class Visitor(ABC):
    def __call__(self, node:Symbol, *args, **kwargs):
        """
        1) 根据 node 的类型动态找到 Visitor.visit_<ClassName>
        2) 如果没有，就降级到 Visitor.generic_visit
        """
        method = getattr(self, 'visit_' + type(node).__name__, self.generic_visit)
        return method(node, *args, **kwargs)

    @abstractmethod
    def generic_visit(self, node:Symbol, *args, **kwargs):
        raise NotImplementedError(f'generic_visit not implemented for {type(self).__name__}')

