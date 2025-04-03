init_board

path: List[str]

children: Dict[str, str]
keys sono hash della board + mossa per arrivarci (path)
values sono le valid moves dalla board children

expand : 

init




TODO:
!usando Move, non stiamo considerando i PASS

Pare non ci siano edge cases in cui un node esplorato non ha figli...

TODO:
la formula usata in _uct_select è diversa da quella proposta da Norelli, andrebbe cambiata

TODO:
ovviamente va fatta la rete neurale e quindi va aggiunto nella classe Node_mcts il valore P

TODO:
da finire la expansion