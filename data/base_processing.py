"""
    @nghia nh
    ---
    Class base for data processing

    This is a common interface processing raw data.
    Giving match output for input of Pipeline's data constructor or classifier
"""


class BaseDataProcessor:
    """
    This form implies that we handle all the data and
    only support getting train - test split from provided indices.

    The process follows:
    reading draw data -> <transform> into required type ->
    get splitting indices(using outside data tool) -> <extract> data from indices.
    """
    def __init__(self):
        """
        Reading raw data and init some basic information
        """
        self._x = None
        self._y = None

        # data number prepares for splitter using a placeholder for actual data
        # (e.g. train-test split indices form sklearn split)
        self._x_number = 0

    @property
    def x_number(self):
        """
        prevent messing this property.
        x_number and y are required for getting split indices.
        :return:
        """
        return self._x_number

    @property
    def y(self):
        """
        prevent messing this property.
        x_number and y are required for getting split indices.
        :return:
        """
        return self._y

    def transform(self):
        """
        This should set the values for self.x and self.y
        as all data features and corresponding labels.
        Also we transform data into required form (e.g. binary, multi-class)
        and overall processing which does not affect the un-touch rule for test data (in future).
        :return:
        """
        return self

    def extract(self, train_indices, test_indices):
        """
        Extract data when we have train and test split indices.
        We should process normalizing here and this mus be processing ONLY on train set
        then apply the same setting for test data.
        :param train_indices: train data indices
        :param test_indices: test data indices
        :return:
        """
        return self._x[train_indices], self._y[train_indices], self._x[test_indices], self._y[test_indices]
