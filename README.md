# STATS-141SL
(Final Version, Now Archived)

Final project repo for STATS 141SL. You can view this readme in a nice format in the following repo link.

repo link: https://github.com/Wangzzzzzzzz/STATS-141SL

## IMPORTANT

Part of the code is written in python (FinalProj_ann.py and reogranize.py). To run those code, you will have to install the related packages in python.

Run the following code using your terminal to install the packages

    pip install numpy==1.18.1 pandas==1.0.1 tqdm==4.42.1 torch==1.4.0 torchvision==0.5.0

If you still run into problem into while runing the python files from the sections below

    ```{bash}
    # run the neural network as a reference to see performance
    # of the logistic regression model
    python3 FinalProj_ann.py
    ```

    ```{bash}
    # run python script to prepare
    # data for the next part
    python3 reorganize.py
    ```

The cause is likely that you have multiple version of python installed on your computer. Since R will run an uninitailized and non-interactive bash session, non of the initialization of terminal session will be run before R execute those commands. We here provides two solutions for you:

#### Solution 1

Use your terminal to run the script `python3 FinalProj_ann.py` and `python3 reorganize.py`. Skip the two section mentioned above and continue running the section below.

#### Solution 2

Add `export PATH=your_python_path:$PATH` at begining of each section. Here "your_python_path" refer to the path of your python command, you can use `which python` to check that (*But notice that you have to exclude "python" at the end*) For instance, if you obtain the following from your terminal, your python path will be `/Users/davidwang/opt/anaconda3/bin`

    > which python
    /Users/davidwang/opt/anaconda3/bin/python

And the sections will become:

    ```{bash}
    # run the neural network as a reference to see performance
    # of the logistic regression model
    export PATH=/Users/davidwang/opt/anaconda3/bin:$PATH
    python3 FinalProj_ann.py
    ```

    ```{bash}
    # run python script to prepare
    # data for the next part
    export PATH=/Users/davidwang/opt/anaconda3/bin:$PATH
    python3 reorganize.py
    ```