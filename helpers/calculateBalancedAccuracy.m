function balanced_accuracy = calculateBalancedAccuracy(actualLabels, predictedLabels)
%function balanced_accuracy = calculateBalancedAccuracy(actualLabels, predictedLabels)
    % Convert inputs to logical arrays for easier processing if they aren't already
    actualLabels = logical(actualLabels);
    predictedLabels = logical(predictedLabels);

    % True Positives, True Negatives, False Positives, False Negatives
    TP = sum(actualLabels == 1 & predictedLabels == 1);
    TN = sum(actualLabels == 0 & predictedLabels == 0);
    FP = sum(actualLabels == 0 & predictedLabels == 1);
    FN = sum(actualLabels == 1 & predictedLabels == 0);

    % Sensitivity (Recall) = TP / (TP + FN)
    if (TP + FN) == 0
        sensitivity = 0; % Handle potential division by zero
    else
        sensitivity = TP / (TP + FN);
    end

    % Specificity = TN / (TN + FP)
    if (TN + FP) == 0
        specificity = 0; % Handle potential division by zero
    else
        specificity = TN / (TN + FP);
    end

    % Balanced Accuracy = (Sensitivity + Specificity) / 2
    balanced_accuracy = (sensitivity + specificity) / 2;
end
