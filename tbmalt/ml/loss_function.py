from tbmalt.ml.module import Calculator
import tbmalt.common.maths as tb_math
from typing import Callable, Optional, Union, Any, Dict, Tuple
from warnings import warn
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor

LossFunction = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]

DataDelegate = Callable[[Calculator, Dataset, Any], Dict[str, Tensor]]

WeightDelegate = Callable[[Calculator, Dataset, Any], Tensor]


def l1_loss(prediction: Tensor, reference: Tensor, weights: Optional[Tensor] = None,
            reduction: Optional[str] = None) -> Tensor:
    """
    Calculates the L1 loss (also known as absolute error) between the prediction and the reference.

    Args:
    - prediction (torch.Tensor): The predicted values.
    - reference (torch.Tensor): The actual values to compare against.
    - weights (torch.Tensor): weights for each system [Optional].
    - reduction (str): the reduction method applied to the output [Optional].

    Returns:
    - torch.Tensor: The calculated L1 loss.
    """
    weights = weights if weights is not None else 1.0
    reduction = reduction if reduction is not None else 'mean'

    if reduction == 'mean':
        loss = torch.abs((prediction - reference) * weights).mean()
    elif reduction == 'sum':
        loss = torch.abs((prediction - reference) * weights).sum()
    return loss


def mse_loss(prediction: Tensor, reference: Tensor, weights: Optional[Tensor] = None,
             reduction: Optional[str] = None) -> Tensor:
    """
    Calculates the mean squared error (squared L2 norm) between the prediction and the reference.

    Args:
    - prediction (torch.Tensor): The predicted values.
    - reference (torch.Tensor): The actual values to compare against.
    - weights (torch.Tensor): weights for each system [Optional].
    - reduction (str): the reduction method applied to the output [Optional].

    Returns:
    - torch.Tensor: The calculated L1 loss.
    """
    weights = weights if weights is not None else 1.0
    reduction = reduction if reduction is not None else 'mean'

    if reduction == 'mean':
        loss = ((prediction - reference) ** 2 * weights).mean()
    elif reduction == 'sum':
        loss = ((prediction - reference) ** 2 * weights).sum()
    return loss


def hellinger_loss(prediction: Tensor, reference: Tensor, weights: Optional[Tensor] = None,
                   reduction: Optional[str] = None) -> Tensor:
    """
    Calculates the hellinger loss between the prediction and the reference.

    Args:
    - prediction (torch.Tensor): The predicted values.
    - reference (torch.Tensor): The actual values to compare against.
    - weights (torch.Tensor): weights for each system [Optional].
    - reduction (str): the reduction method applied to the output [Optional].

    Returns:
    - torch.Tensor: The calculated L1 loss.
    """
    weights = weights if weights is not None else 1.0
    reduction = reduction if reduction is not None else 'mean'

    if reduction == 'mean':
        loss = (tb_math.hellinger(prediction, reference).unsqueeze(-1) * weights).mean()
    elif reduction == 'sum':
        loss = (tb_math.hellinger(prediction, reference).unsqueeze(-1) * weights).sum()
    return loss


class Loss:
    """Component for extracting prediction/reference data & computing losses.

    This class facilitates the evaluation of losses by delegating the
    responsibility for fetching prediction data, reference data, and optional
    system weights to user-defined functions. The flexibility in defining
    custom loss functions for different properties and their respective weights
    allows for tailored optimisation strategies.

    This entity is fairly simple in operation as it contains very little in the
    way of active logic. It is primarily intended as a means of clearing up the
    main training loop, as it abstracts some of the tedious & repetitive code.

    Attributes:
        prediction_data_delegate: A delegate function tasked with fetching
            predicted data from a given calculator entity.
        reference_data_delegate: A delegate function responsible for obtaining
            reference from a dataset.
        system_weight_delegate: An optional delegate function that provides
            weights for each system present in a batch. This allows for some
            molecules to have a greater or lesser influence on the loss. By
            default, system specific weighting is disabled.
        loss_functions: A single loss function, or a dictionary of such keyed
            by property names, used for computing individual losses. This will
            default to the L1 loss. If using a dictionary, the key `default`
            can be used to specify what loss function should be used for
            properties whose loss functions are not explicitly specified.
            [DEFAULT=l1_loss]
        loss_weights: A dictionary mapping property names to their respective
            weights in the final loss calculation. If not provided, a uniform
            weighting is assumed.
        reduction: The reduction method used to obtain output, which has the
            option of 'mean' or 'sum'. By default, 'mean' will be applied.

    Note:
        It is expected that the prediction & reference data delegate functions
        will each accept three parameters: `calculator`, `dataset`, & `**kwargs`.
        These functions must return a dictionary containing the properties to
        be utilised in the computation of the total loss. Runtime warnings will
        be issued if and when key dependencies are encountered. That is to say
        if there are keys found in one dictionary that are not in the other.

        Optionally, a system weight delegate method may be provided through the
        ``system_weight_delegate`` argument. This method should return a torch
        tensor that specifies a weight for each system within the current batch,
        adhering to the same calling convention as the prediction and reference
        data delegate functions.

        The mechanism for computing the component losses tied to each property
        can be defined using the ``loss_functions`` argument. This allows for
        the assignment of distinct loss functions to different properties by
        employing a dictionary. Additionally, property-specific weights may be
        designated via the ``loss_weights`` attribute, facilitating the
        calculation of the final loss as a weighted sum of all individual
        losses rather than a mere aggregation.


    Example:
        The class is instantiated with delegate functions for data retrieval &
        optionally system weighting, alongside a specification of loss functions
        and their weights.

        >>> def example_prediction_delegate(calculator, dataset, **kwargs):
        >>>     predictions = dict()
        >>>     predictions["fermi_energy"] = calculator.fermi_energy
        >>>     predictions["mulliken"] = calculator.q_final
        >>>     return predictions
        >>>
        >>> def example_reference_delegate(calculator, dataset, **kwargs):
        >>>     references = dict()
        >>>     references["fermi_energy"] = dataset.data["fermi_energy"]
        >>>     references["mulliken"] = dataset.data["mulliken_population"]
        >>>     return references
        >>>
        >>> loss_entity = Loss(example_prediction_delegate,
        >>>                    example_reference_delegate)

        It is then called with a calculator & dataset to compute the total and
        individual losses, facilitating the optimisation of complex systems
        with multiple properties.

        >>> some_calculator(some_geometry, some_orbitalinfo)
        >>> total_loss, raw_losses = loss_entity(some_calculator, some_dataset)


    """

    def __init__(
            self, prediction_data_delegate: DataDelegate,
            reference_data_delegate: DataDelegate,
            system_weight_delegate: Optional[WeightDelegate] = None,
            loss_functions: Union[LossFunction, Dict[str, LossFunction]] = l1_loss,
            loss_weights: Dict[str, float] = None,
            reduction: Optional[str] = None):


        self.prediction_data_delegate = prediction_data_delegate
        self.reference_data_delegate = reference_data_delegate
        self.system_weight_delegate = system_weight_delegate

        # If a loss function dictionary has been provided then ensure that it
        # contains a default loss function.
        if isinstance(loss_functions, dict):
            self.loss_functions = loss_functions
            if "default" not in loss_functions:
                loss_functions["default"] = l1_loss

        # If a single function has been provided, then create a loss function
        # dictionary with this function as its default loss function.
        elif isinstance(loss_functions, Callable):
            self.loss_functions = dict()
            self.loss_functions["default"] = loss_functions

        # If something other than a dictionary of function was provided then
        # just raise an exception.
        else:
            raise ValueError( "The `loss_functions` argument may only be a "
                              "dictionary or a callable")

        self.loss_weights = loss_weights if loss_weights else dict()

        # Choose the reduction method
        if reduction is not None:
            if reduction in ['mean', 'sum']:
                self.reduction = reduction
            else:
                raise ValueError( "The `reduction` argument may only be 'mean' or "
                                  "'sum'.")
        else:
            self.reduction = reduction


    def __call__(self, calculator: Calculator, dataset: Dataset, **kwargs
                 ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Evaluate the total loss.

        This method calls out to the class delegate functions to retrieve both
        predicted and reference values for a collection of properties. Losses
        are then computed for each individual property and a total final
        weighted sum is returned.

        In addition to the total value, the individual unweighted (raw) loss
        components are also returned for the sake of completeness.

        Arguments:
            calculator: An initialised calculator object that can be
                interrogated to populate the predictions dictionary.
            dataset: A data-set entity from which reference data can be
                extracted to populate the references dictionary.
            **kwargs: Keyword arguments are not consumed by this method, but
                are instead directly passed through to the delegate functions.

        Returns:
            total_loss: The total loss as a weighted sum of the individual
                loss components.
            raw_loss: A dictionary of all the individual, unweighted, loss
                components.

        Notes:
            It is imperative to ensure that the supplied calculator has already
            been initialised prior to calling this method on it. If this is not
            the case then the calculator will likely be missing vital pieces of
            information. Specifically, it is expected that the user, or another
            part of the code, has already supplied the calculator with all of
            the necessary components to operate e.g. `Geometry`, `OrbitalInfo`,
            etc. This method will never perform the `Calculator(Geometry, ...)`
            call itself.

        """

        # Gather the predicted values by calling the prediction delegate
        predictions = self.prediction_data_delegate(calculator, dataset, **kwargs)

        # Repeat for the reference values
        references = self.reference_data_delegate(calculator, dataset, **kwargs)

        # Get system specific weights if a delegate function was provided
        system_weights = (self.system_weight_delegate(calculator, dataset)
                          if self.system_weight_delegate else None)

        # Get reduction method
        reduction = self.reduction

        # Identify any keys that present in one data dictionary but not the
        # other. This helps to catch typos or truant properties.
        missing_in_predictions = set(references.keys()) - set(predictions.keys())
        missing_in_references = set(predictions.keys()) - set(references.keys())
        if missing_in_predictions or missing_in_references:
            message = "Missmatch detected between prediction & reference " \
                      "data key sets:"

            if missing_in_predictions:
                message += f"\n\t - Keys absent from predictions: " \
                           f"{missing_in_predictions}"

            if missing_in_references:
                message += f"\n\t - Keys absent from predictions: " \
                           f"{missing_in_references}"
            warn(message)

        raw_losses = dict()

        # Loop over each of the predicted properties as returned by the prediction
        # delegate function.
        for name, prediction in predictions.items():

            # Retrieve the reference value for this property
            reference = references[name]

            # Check if the tensor size of prediction and reference identical
            check = prediction.shape == reference.shape
            assert check, 'prediction & reference size mismatch found for ' + name

            # Check if a custom loss function has been explicitly defined for
            # this property. If not, then fallback to the default function.
            loss_function = self.loss_functions.get(
                name, self.loss_functions["default"])

            # Compute the loss value for this property, taking into account the
            # individual system weights if applicable.
            raw_losses[name] = loss_function(
                prediction, reference, weights=system_weights, reduction=reduction)

        # Compute the total loss as the weighted sum of the raw loss values. A
        # weight value of one is used in situations where property specific
        # loss values are not explicitly provided.
        total_loss = sum([loss * self.loss_weights.get(name, 1)
                          for name, loss in raw_losses.items()])

        # Return both the total and raw loss values
        return total_loss, raw_losses
