function [f_diff] = proxGrad_backtrack(ratings,...
    labels,groupSizes, iters, lambda, b)
    [~, features] = size(ratings);
    fVals = zeros(iters, 1);
    currBeta = zeros(features+1,1);
    allBetas = zeros(iters, features+1);
    fVals_allIters = [];
    allIters = 0;
    f_optimal = 306.476;
    
    for i = 1:iters
        penalty = 0;
        interceptInd = 1;
        for k = 1:length(groupSizes)
            if k == 1
                startInd = interceptInd + 1;
            else
                startInd = interceptInd + sum(groupSizes(1:k-1))+1;
            end
            endInd = startInd + groupSizes(k) - 1;
            penalty = penalty + ...
            sqrt(groupSizes(k))*norm(currBeta(startInd:endInd),2);
        end
        fVals(i) = evalLoss(currBeta, ratings, labels) + lambda*penalty;
        grad = evalGradient(currBeta, ratings, labels);
        t = 1;
        G = (currBeta - softThresh_groupLasso_binary(currBeta - ...
        t*grad, lambda*t, groupSizes))./t;
        lossG = evalLoss(currBeta - t*G, ratings, labels);
        loss = evalLoss(currBeta, ratings, labels);
        while lossG > loss - t*grad'*G + 0.5*t*sum(G.^2)
            t = b*t;
            G = (currBeta - softThresh_groupLasso_binary(currBeta - ...
            t*grad, lambda*t, groupSizes))./t;
            lossG = evalLoss(currBeta - t*G, ratings, labels);
            allIters = allIters + 1;
            fVals_allIters(allIters) = fVals(i);
        end
        currBeta = softThresh_groupLasso_binary(currBeta - ...
        t*grad, lambda*t, groupSizes);
        allBetas(i,:) = currBeta;
    end
    
    f_diff = fVals - f_optimal;
end

function [output] = sigm(input)
    output = 1./(1 + exp(-input));
end

function [gradient] = evalGradient(evalPoints, ratings, labels)
    samples = length(labels);
    gradient = [ones(1,samples); ratings']*...
    (sigm([ones(samples,1), ratings]*evalPoints) - labels);
end

function [newBeta] = softThresh_groupLasso_binary(beta, thresh, groupSizes)
    newBeta = zeros(size(beta));
    interceptInd = 1;
    % we don?t threshold the intercept weight
    newBeta(interceptInd) = beta(interceptInd);
    for i = 1:length(groupSizes)
    % calculate where the group features start and end
        if i == 1
            startInd = interceptInd + 1;
        else
            startInd = interceptInd + sum(groupSizes(1:i-1))+1;
        end
        endInd = startInd + groupSizes(i) - 1;
        % threshold
        if norm(beta(startInd:endInd),2) > thresh*sqrt(groupSizes(i))
            newBeta(startInd:endInd) = beta(startInd:endInd) - ...
            sqrt(groupSizes(i))*thresh.*beta(startInd:endInd)./...
            norm(beta(startInd:endInd),2);
        else
            newBeta(startInd:endInd) = zeros(length(startInd:endInd),1);
        end
    end
end

function [loss] = evalLoss(evalPoints, ratings, labels)
    samples = length(labels);
    loss = -labels'*[ones(samples,1), ratings]*evalPoints - ...
    ones(1,samples)*log(sigm(-[ones(samples,1), ratings]*evalPoints));
end