import unittest
import counting
from functools import partial

class TestCounting(unittest.TestCase):
    func = partial(counting.count_coins, tests=True)
    def test_coins_amm(self):
        """Sprawdza czy monety liczone są poprawnie"""

        for i in range(1, 14):
            self.assertEqual(self.func(f'img/monety{i}.jpg'), 18)
        
        self.assertEqual(self.func('img/monety14.jpg'), 28)
        self.assertEqual(self.func('img/monety15.jpg'), 9)
    
    def test_incorrect_name(self):
        """Sprawdza czy funkcja zwraca błąd gdy nie ma takiego pliku"""
        self.assertRaises(ValueError, self.func, 'img/nie_istnieje.jpg')
        self.assertRaises(ValueError, counting.get_coin_size_in_pixels, 'img/nie_istnieje.jpg')

    def test_incorrect_type(self):
        """Sprawdza czy funkcja zwraca błąd gdy typ danych jest niepoprawny"""
        self.assertRaises(TypeError, self.func, True)
        self.assertRaises(TypeError, self.func, 5)
        self.assertRaises(TypeError, self.func, 5.5)
    
        self.assertRaises(TypeError, counting.get_coin_size_in_pixels, True)
        self.assertRaises(TypeError, counting.get_coin_size_in_pixels, 5)
        self.assertRaises(TypeError, counting.get_coin_size_in_pixels, 5.5)