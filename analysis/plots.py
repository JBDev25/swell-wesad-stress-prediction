import matplotlib

matplotlib.use('pdf')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils, paths
import subprocess


marker_size = 4
width = 3.487
height = width / 1.618

def set_plot_style_():
    plt.style.use('seaborn-paper')
    settings = {
        'lines.linewidth': 0.5,
        # grid
        'grid.color': '0.3',
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,

        # ax.grid(linestyle=':', linewidth='0.5', alpha=0.5)
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.grid.axis': 'both',
        # patch edges
        'patch.edgecolor': 'black',
        'patch.force_edgecolor': True,
        'patch.linewidth': 0.5,

    }
    matplotlib.rcParams.update(settings)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)
    plt.rc('legend', fontsize=6)



def save_figure(file_name, formats=None, plot_folder=None, fig=None):
    fig.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0)
    if formats is None:
        formats = ['png', 'eps', 'svg', 'pdf']
    if plot_folder is None:
        plot_folder = paths.plots_directory()
    for fmt in formats:
        out_dir = os.path.join(plot_folder, fmt)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plot_file = os.path.join(out_dir, file_name + "." + fmt)
        if fig is not None:
            fig.savefig(plot_file, format=fmt, bbox_inches='tight', pad_inches=0.0, optimize=True, dpi=600,
                        transparent=True)
        else:
            plt.savefig(plot_file, format=fmt, bbox_inches='tight', pad_inches=0.0, optimize=True, dpi=600,
                        transparent=True)
        # Trim the extra space
        #Not necessary with matplotlib settings
        #trim_plot(plot_file=plot_file)
    #plt.close()


def trim_plot(plot_file):

    file_name, extension = os.path.splitext(plot_file)
    in_dir = os.path.dirname(plot_file)
    temp_file = os.path.join(in_dir, file_name+"-temp"+extension)

    os.rename(plot_file, temp_file)
    # trim it
    if extension == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                  (temp_file, plot_file), shell=True)
    elif extension == 'png':
        subprocess.call('convert %s -trim %s' %
                  (temp_file, plot_file), shell=True)
    elif extension == 'pdf':
        #subprocess.call('pdfcrop %s %s' % (temporary_file, original_file), shell=True)
        #subprocess.call('pdfcrop %s %s' % (temp_file, plot_file), shell=True)
        os.system("pdfcrop %s %s" % (temp_file, plot_file))
    #delete the temporary file
    #os.remove(temp_file)

def plot_calibration_classification_result(ax, dataset, signal, model_name):
    model_name = model_name+"Classifier"
    set_plot_style_()
    in_dir = os.path.join(paths.result_directory(), "tables", "classification", dataset, signal, "calibration")
    file_name = dataset + "-" + signal + "-" + "classification" + ".csv"
    data = pd.read_csv(os.path.join(in_dir, model_name, file_name), index_col=0)
    accuracy = (data.loc["Accuracy"] * 100).tolist()
    precision = (data.loc["Precision"] * 100).tolist()
    calibration_samples = data.columns.values.tolist()
    calibration_samples = [int(x) for x in calibration_samples]
    ax.plot(calibration_samples, precision, marker='o', markersize=marker_size, linestyle='-', markevery=1,
            color="#e41a1c", label='precision (%)')
    ax.plot(calibration_samples, accuracy, marker='o', markersize=marker_size, linestyle='-', markevery=1,
            color="#377eb8", label='accuracy (%)')
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), ncol=1, fancybox=True, shadow=False, framealpha=1.0)
    leg.get_frame().set_edgecolor('w')
    ax.set_yticks(np.arange(70, 101, 10))
    ax.set_xlim([-3, 101])
    ax.set_ylabel('classification', fontsize=6)
    sns.despine(ax=ax, offset=0, trim=False)
    ax.grid()


def plot_calibration_regression_result(ax, dataset, signal, model_name):
    model_name= model_name+"Regressor"
    set_plot_style_()
    in_dir = os.path.join(paths.result_directory(), "tables/regression/", dataset, signal, "calibration")
    file_name = dataset + "-" + signal + "-" + "regression" + ".csv"
    data = pd.read_csv(os.path.join(in_dir, model_name, file_name), index_col=0)
    rmse = data.loc["RMSE"].tolist()
    mae = data.loc["MAE"].tolist()
    calibration_samples = data.columns.values.tolist()
    calibration_samples = [int(x) for x in calibration_samples]

    ax.plot(calibration_samples, mae, marker='o', markersize=marker_size, linestyle='-', markevery=1,
            color="#4daf4a", label='mean absolute error')
    ax.plot(calibration_samples, rmse, marker='o', markersize=marker_size, linestyle='-', markevery=1,
            color="#984ea3", label='root mean square error')

    leg = ax.legend(loc='upper right', ncol=1,bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=False, framealpha=1.0)
    leg.get_frame().set_edgecolor('w')
    # ax.set_xticks(np.arange(-10, 101, 10))
    ax.tick_params(axis='x', which='minor', direction='out', bottom=True, length=10)
    ax.set_xlim([-3, 101])

    ax.set_yticks(np.arange(0, 1.26, 0.25))
    ax.set_xticks(np.arange(0, 101, 10))
    plt.xticks(np.arange(0, 101, 10))
    ax.set_ylim([-0.1, 1.25])
    sns.despine(ax=ax, offset=0, trim=False)
    ax.set_ylabel('regression', fontsize=6)
    ax.set_xlabel('calibration samples per subject', fontsize=6)
    ax.grid()
    plt.grid()



def plot_calibration_result(dataset, signal, model_name):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    plot_calibration_classification_result(ax=axes[0], dataset=dataset, signal=signal,model_name=model_name)
    plot_calibration_regression_result(ax=axes[1], dataset=dataset, signal=signal, model_name=model_name)
    out_dir = paths.ensure_directory_exists(
        os.path.join(paths.plots_directory(), dataset, signal, "model calibration", model_name))
    file_name = dataset + "-" + signal + "-" + "calibration"
    fig.set_size_inches(width, height)
    plt.grid()
    save_figure(file_name=file_name, plot_folder=out_dir, formats=None, fig=fig)


def plot_generic_vs_person_specific_model(dataset, signal, model_name):
    subjects = utils.get_subjects_ids(dataset=dataset)
    set_plot_style_()
    def plot_classification_results(ax, dataset, signal):
        clf_model_name = model_name+"Classifier"
        file_name = dataset + "-" + signal + "-" + "classification"
        pers_spec_df = pd.read_excel(os.path.join(paths.result_directory(), "tables", "classification",
                                                dataset, signal, "person-specific",
                                                  clf_model_name, file_name+".xlsx"), index_col="Fold")
        generic_df = pd.read_csv(
            os.path.join(paths.result_directory(), "tables", "classification", dataset, signal, "generic", clf_model_name,
                         file_name+".csv"), index_col="valuation metrics")

        pers_spec_acc = (pers_spec_df.loc["mean"]).tolist()
        generic_acc = (generic_df.loc["Accuracy"] * 100).tolist()
        ax.plot(subjects, pers_spec_acc, marker='o', markersize=marker_size, linestyle='-',
                color="#ca0020", label='person-specific model')
        ax.plot(subjects, generic_acc, marker='o', markersize=marker_size, linestyle='-',
                color="#6a3d9a", label='generic model')
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=False,
                        framealpha=1.0)

        leg.get_frame().set_edgecolor('w')
        ax.set(ylabel='accuracy (%)')
        ax.set_yticks(np.arange(20, 101, 20))
        sns.despine(ax=ax, offset=0, trim=False)

    def plot_regression_results(ax, dataset, signal):

        file_name = dataset + "-" + signal + "-" + "regression"
        clf_model_name = model_name + "Regressor"
        pers_spec_df = pd.read_excel(
            os.path.join(paths.result_directory(), "tables", "regression", dataset, signal,
                         "person-specific", clf_model_name,file_name+".xlsx"), sheet_name=None, index_col="cv fold")
        pers_spec_rmse =pers_spec_df["rmse"].loc["mean"].tolist()

        generic_df = pd.read_csv(os.path.join(paths.result_directory(), "tables", "regression",
                                              dataset, signal, "generic", clf_model_name,file_name+".csv"))
        ###
        cols = list(generic_df)
        cols = [x for x in cols if x not in ["min", "mean", "max", "std"]]
        generic_df = generic_df[cols]

        generic_rmse = generic_df["RMSE"].tolist()
        ax.plot(subjects, pers_spec_rmse, marker='o', markersize=marker_size, linestyle='-',
                color="#ca0020",
                label='person-specific model')
        ax.plot(subjects, generic_rmse, marker='o', markersize=marker_size, linestyle='-',
                color="#6a3d9a",
                label='generic model')
        ax.set_yticks(np.arange(0, 41, 10))
        ax.set_ylim([-3, 40])
        ax.set(ylabel='RMS error')
        ax.set(xlabel='subject id')
        sns.despine(ax=ax, offset=0, trim=False)

    out_dir = paths.ensure_directory_exists(
        os.path.join(paths.plots_directory(), dataset, signal, "model comparision", model_name))
    print(out_dir)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

    plot_classification_results(ax=axes[0], dataset=dataset, signal=signal)
    plot_regression_results(ax=axes[1], dataset=dataset, signal=signal)
    file_name = dataset + "-" + signal + "-" + "generic-vs-pers-spect"
    fig.set_size_inches(width, height)
    save_figure(file_name=file_name, plot_folder=out_dir, formats=None, fig=fig)
if __name__ == "__main__":
    plot_generic_vs_person_specific_model(dataset="swell", signal="hrv", model_name="ExtraTrees")
