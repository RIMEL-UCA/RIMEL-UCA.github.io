magnitude = [x for x in range(2,11)]
magnitude.extend([10,10,10,11])
magnitude
suits = ['Spade','Club','Hearts','Diamond']
Rank =  ['Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','king','Ace']
values = {keys : value for keys,value in zip(Rank,magnitude)}
class Card:
    def __init__(self,suit,rank):
        self.suit = suit
        self.rank = rank
        self.value = values[self.rank]
    def __str__(self):
        return '{} of {}'.format(self.rank,self.suit)
import random
class Deck:
    def __init__(self):
        self.all_cards = []
        for i in suits:
            for j in Rank:
                self.all_cards.append(Card(i,j))
    def shuffle(self):
        random.shuffle(self.all_cards)
    def deal(self):
        return self.all_cards.pop()
class Hand:
    def __init__(self):
        self.cards = []
        self.value = 0          
        self.aces = 0
    def add_card(self,card):
        self.cards.append(card)
        self.value = self.value+ card.value
        if card.rank == "Ace":
            self.aces = self.aces+1
            
    def adjust_for_ace(self):
        while self.value>21 and self.aces:
            self.value = self.value-10
            self.aces = self.aces-1
test_deck = Deck()
test_deck.shuffle()

test_player = Hand()
test_player.add_card(test_deck.deal())
test_player.add_card(test_deck.deal())
test_player.value
class Chips:
    def __init__(self):
        self.total = 200
        self.bet = 0
    def win_bet(self):                     
        self.total = self.total+self.bet
    def lose_bet(self):
        self.total = self.total-self.bet
def take_bet(chips):
    while True:
        try:
            chips.bet = int(input('Enter your bet : '))
        except:
            print('Kindly enter a integer value')
        else:
            if chips.bet>chips.total:
                print('Not enough chips! You have {} '.format(chips.total))
            else:
                break
def hit(deck,hand):
    hand.add_card(deck.deal())
    hand.adjust_for_ace()
def hit_or_stand(deck,hand):
    global playing
    while True:
        x= input('hit or stand? H|S :').upper()
        if x == 'H':                                      #???????
            hit(deck,hand)
        elif x=='S':
            print('Player chooses to stand')
            playing = False
        else:
            print('Please enter either H or S only')
            continue
        
        break
from IPython.display import clear_output as clr
def show_some(player,dealer):
    clr()
    print('Players value : {}'.format(player.value))
    print('Players card',end=': ')#hand classess
    for i in player.cards:
        print(i,end=',')
    # For dealer
    print('\n')
    print('Dealers single card:',end=' ')
    print(dealer.cards[0])
    print('\n')

def show_all(player,dealer):
    print('Players value : {}'.format(player.value))
    print('Players card',end=': ')
    for i in player.cards:
        print(i,end=',')
    print('\n')
    print('Dealers value :{}'.format(dealer.value))
    print('DEALER CARDS:',end =' ')
    for i in dealer.cards:
        print(i,end=',')
    print('\n')
def player_busts(player,dealer,chips):
    print('BUST PLAYER!')
    chips.lose_bet()
def player_wins(player,dealer,chips):
    print('Player wins')
    chips.win_bet()
def dealer_wins(player,dealer,chips):
    print('Dealer wins')
    chips.lose_bet()
def dealer_busts(player,dealer,chips):
    print('Dealer busts,player win')
    chips.win_bet()
def push(player,dealer):
    print('TIE!')
def replay():
    choice= ''
    while choice not in ['Y','N']:
        choice = input('Do you want to play again Y|N: ').upper()
    return choice=='Y'
playing = True
restart = True
while restart:
    print('Ready to Play Black Jack')
    deck = Deck()
    deck.shuffle()
    dealer= Hand()
    player = Hand()
    
    for i in range(2):
        dealer.add_card(deck.deal())
        player.add_card(deck.deal())
        
    player_chips = Chips()
    take_bet(player_chips)
    show_some(player,dealer)
    
    while playing:
        print('Player choose!')
        hit_or_stand(deck,player)
        show_some(player,dealer)
        if player.value >21:
            player_busts(player,dealer,player_chips)
            break
    if player.value <=21:
        while dealer.value <17:
            print('Dealer choose!')
            hit(deck,dealer)
        show_all(player,dealer)
        if dealer.value>21:
            dealer_busts(player,dealer,player_chips)
        elif dealer.value>player.value:
            dealer_wins(player,dealer,player_chips)
        elif dealer.value == player.value:
            push(player,dealer)
        else:
            player_wins(player,dealer,player_chips)
    
    print('\n')
    print('player you have {} total chips'.format(player_chips.total))
    restart = replay()
