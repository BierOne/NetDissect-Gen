"""Functions for computing top-activating images in vision models.

Callers should generally only use `discriminative` or `generative`, depending
on what type of model you are computing exemplars for. If you're feeling
edgy, you can instead call `compute` directly, which only requires you
pass an arbitrary function that computes activations given images.
"""
import pathlib
import shutil, os, inspect
from typing import Any, Callable, Optional, Sequence, Tuple

from src.deps import netdissect
from src.deps.ext.netdissect import imgviz
from src.deps.netdissect import (imgsave, nethook, pbar, renormalize, pidfile,
                                 upsample, runningstats, tally, segmenter, bargraph)

from src.exemplars import transforms
from src.utils import env
from src.utils.typing import Device, Layer, PathLike, TensorPair

import numpy
import torch
from torch import nn
from torch.utils import data

ActivationStats = Tuple[runningstats.RunningTopK, runningstats.RunningQuantile]


def load_segmenter(segmenter_name='netpqc'):
    '''Loads the segementer.'''
    all_parts = ('p' in segmenter_name)
    quad_seg = ('q' in segmenter_name)
    textures = ('x' in segmenter_name)
    colors = ('c' in segmenter_name)
    segmodels = []
    segmodels.append(segmenter.UnifiedParsingSegmenter(segsizes=[256],
            all_parts=all_parts,
            segdiv=('quad' if quad_seg else None)))
    if textures:
        segmenter.ensure_segmenter_downloaded('data/segmodel', 'texture')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="texture", segarch=("resnet18dilated", "ppm_deepsup")))
    if colors:
        segmenter.ensure_segmenter_downloaded('data/segmodel', 'color')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="color", segarch=("resnet18dilated", "ppm_deepsup")))
    if len(segmodels) == 1:
        segmodel = segmodels[0]
    else:
        segmodel = segmenter.MergedSegmenter(segmodels)
    seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
    segcatlabels = segmodel.get_label_and_category_names()[0]
    return segmodel, seglabels, segcatlabels


def compute(compute_topk_and_quantile: Callable[..., TensorPair],
            compute_activations: imgviz.ComputeActivationsFn,
            dataset: data.Dataset,
            units: Optional[Sequence[int]] = None,
            k: int = 5,
            quantile: float = 0.99,
            output_size: int = 224,
            batch_size: int = 128,
            image_size: Optional[int] = None,
            renormalizer: Optional[renormalize.Renormalizer] = None,
            upfn: Optional[Any] = None,
            num_workers: int = 30,
            results_dir: Optional[PathLike] = None,
            viz_dir: Optional[PathLike] = None,
            tally_cache_file: Optional[PathLike] = None,
            masks_cache_file: Optional[PathLike] = None,
            save_results: bool = True,
            save_viz: bool = True,
            clear_cache_files: bool = False,
            clear_results_dir: bool = False,
            clear_viz_dir: bool = False,
            display_progress: bool = True) -> ActivationStats:
    """Find and visualize the top-activating images for each unit.

    Top-activating images are found using netdissect [Bau et al., 2017].
    This function just forwards to the library. We do not explicitly take a
    model as input but rather two blackbox functions which take dataset batches
    as input and return unit activations as output.

    Args:
        compute_topk_and_quantile (Callable[..., TensorPair]): Function taking
            dataset batch as input and returning tuple of (1) pooled unit
            activations with shape (batch_size, units), and (2) unpooled unit
            activations with shape (*, units).
        compute_activations (imgviz.ComputeActivationsFn): Function taking
            dataset batch as input and returning activations with shape
            (batch_size, channels, *) and optionally the associated images
            of shape (batch_size, channels, height, width).
        dataset (data.Dataset): Dataset to compute activations on.
        units (Optional[Sequence[int]], optional): Only compute exemplars for
            these units. Defaults to all units.
        k (int, optional): Number of top-activating images to save.
            Defaults to 15.
        quantile (float, optional): Activation quantile to use when visualizing
            top images. Defaults to 0.99 (top 1% of activations).
        batch_size (int, optional): Max number of images to send through the
            model at any given time. Defaults to 128.
        output_size (int, optional): Top images and masks will be resized to be
            square in this size. Defaults to 224.
        image_size (Optional[int], optional): Expected size of dataset images.
            If not set, will attempt to infer from the dataset's `transform`
            property. If dataset does not have `transform`, this function will
            self-destruct. Defaults to None.
        renormalizer (Optional[renormalize.Renormalizer], optional): NetDissect
            renormalizer for the dataset images. If not set, NetDissect will
            attempt to infer it from the dataset's `transform` property.
            Defaults to None.
        num_workers (int, optional): When loading or saving data in parallel,
            use this many worker threads. Defaults to 30.
        results_dir (Optional[PathLike], optional): Directory to write
            results to. Defaults to 'results/exemplars'.
        viz_dir (Optional[PathLike], optional): Directory to write top image
            visualizations to (e.g., individual png images, lightbox, etc.).
            Defaults to f'{results_dir}/viz'.
        tally_cache_file (Optional[PathLike], optional): Write intermediate
            results for tally step to this file. Defaults to None.
        masks_cache_file (Optional[PathLike], optional): Write intermediate
            results for determining top-k image masks to this file.
            Defaults to None.
        save_results (bool, optional): If set, save exemplars and metadata
            to results_dir. Otherwise, save nothing (viz may still be saved,
            see `save_viz` arg). Defaults to True.
        save_viz (bool, optional): If set, save individual masked images to
            `viz_dir`. Otherwise, `viz_dir` will not be used. Defaults to True.
        clear_cache_files (bool, optional): If set, clear existing cache files
            with the same name as any of the *_cache_file arguments to this
            function. Useful if you want to redo all computation.
            Defaults to False.
        clear_results_dir (bool, optional): If set, clear the results_dir if
            it exists. Defaults to False.
        clear_viz_dir (bool, optional): If set, clear the viz_dir if
            it exists. Defaults to False.
        display_progress (bool, optional): If True, display progress bar.
            Defaults to True.

    Raises:
        ValueError: If `k` or `quantile` are invalid.

    Returns:
        ActivationStats: The top-k and quantile statistics for every unit.

    """
    if units is not None and not units:
        raise ValueError('when setting `units`, must provide >= 1 unit')
    if k < 1:
        raise ValueError(f'must have k >= 1, got k={k}')
    if quantile <= 0 or quantile >= 1:
        raise ValueError('must have quantile in range (0, 1), '
                         f'got quantile={quantile}')
    if image_size is None and not hasattr(dataset, 'transform'):
        raise ValueError('dataset has no `transform` property so '
                         'image_size= must be set')

    if results_dir is None:
        results_dir = env.results_dir() / 'exemplars'
    if not isinstance(results_dir, pathlib.Path):
        results_dir = pathlib.Path(results_dir)

    if viz_dir is None:
        viz_dir = results_dir / 'viz'
    if not isinstance(viz_dir, pathlib.Path):
        viz_dir = pathlib.Path(viz_dir)

    if tally_cache_file is None:
        tally_cache_file = pidfile.exclusive_dirfn(results_dir)
    else:
        tally_cache_file = pidfile.exclusive_dirfn(tally_cache_file)


    # Clear cache files if requested.
    if clear_cache_files:
        for cache_file in (tally_cache_file, masks_cache_file):
            if cache_file is None:
                continue
            cache_file = pathlib.Path(cache_file)
            if cache_file.exists():
                cache_file.unlink()

    # Clear results and viz directories if requested.
    for save, clear, directory in (
        (save_results, clear_results_dir, results_dir),
        (save_viz, clear_viz_dir, viz_dir),
    ):
        if not save:
            continue
        if clear and directory.exists():
            shutil.rmtree(results_dir)
        directory.mkdir(exist_ok=True, parents=True)

    # Handle case where specific units to be dissected.
    if units is not None:
        units = sorted(units)
        numpy.save(f'{results_dir}/units.npy', numpy.array(units))

        def _compute_topk_and_quantile(*args: Any) -> TensorPair:
            pooled, activations = compute_topk_and_quantile(*args)
            return pooled[:, units], activations[:, units]

        def _compute_activations(*args: Any) -> imgviz.Activations:
            outputs = compute_activations(*args)
            if not isinstance(outputs, torch.Tensor):
                activations, images = outputs
                return activations[:, units], images
            else:
                return outputs[:, units]
    else:
        _compute_topk_and_quantile = compute_topk_and_quantile
        _compute_activations = compute_activations

    sample_size = 1000
    # sample_size = len(dataset)
    # We always compute activation statistics across dataset.
    if display_progress:
        pbar.descnext('tally activations')
    topk, rq = tally.tally_topk_and_quantile(_compute_topk_and_quantile,
                                             dataset,
                                             k=k,
                                             sample_size=sample_size,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             cachefile=tally_cache_file('topk_rq.npz'))

    # Now compute top images and masks for the highest-activating pixels if
    # there is any reason to do so.
    masked, images, masks = None, None, None
    if save_results or save_viz or masks_cache_file is not None:
        if display_progress:
            pbar.descnext('compute top images')
        levels = rq.quantiles(quantile).reshape(-1)
        viz = imgviz.ImageVisualizer(output_size,
                                     image_size=image_size,
                                     renormalizer=renormalizer,
                                     source=dataset,
                                     level=levels)

        unit_images = viz.masked_images_for_topk(
            _compute_activations,
            dataset,
            topk,
            k=k,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            cachefile=tally_cache_file('top%dimages.npz' % k))

    if save_viz:
        pbar.descnext('saving images')
        imgsave.save_image_set(unit_images, tally_cache_file('image/unit%d.jpg'),
                               sourcefile=tally_cache_file('top%dimages.npz' % k))

    #     masked, images, masks = viz.individual_masked_images_for_topk(
    #         _compute_activations,
    #         dataset,
    #         topk,
    #         k=k,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         cachefile=masks_cache_file)
    #
    # if save_viz:
    #     assert masked is not None
    #     # Now save the top images with the masks overlaid. We save each image
    #     # individually so they can be visualized and/or shown on MTurk.
    #     imgsave.save_image_set(masked,
    #                            f'{viz_dir}/unit_%d/image_%d.png',
    #                            sourcefile=masks_cache_file)
    #
    #     # The lightbox lets us view all the masked images at once. Handy!
    #     lightbox_dir = pathlib.Path(__file__).parents[1] / 'deps'
    #     lightbox_file = lightbox_dir / 'lightbox.html'
    #     for unit in range(len(masked)):
    #         unit_dir = viz_dir / f'unit_{unit}'
    #         unit_lightbox_file = unit_dir / '+lightbox.html'
    #         shutil.copy(lightbox_file, unit_lightbox_file)



    # Compute IoU agreement between segmentation labels and every unit
    # Grab the 99th percentile, and tally conditional means at that level.
    # export CUDA_HOME=/usr/local/cuda-11.1/
    seg_name = 'netpqxc'
    iou_threshold = 0.04
    level_at_99 = rq.quantiles(quantile).cuda()[None,:,None,None]
    segmodel, seglabels, segcatlabels = load_segmenter('netpqc')
    renormalizer = renormalize.renormalizer(dataset, target='zc')
    def compute_conditional_indicator(*inputs: Any):
        hiddens, images = compute_activations(*inputs)
        images = renormalizer(images)
        seg = segmodel.segment_batch(images, downsample=4)
        # print(images.shape, hiddens.shape)
        hacts = upfn(hiddens)
        iacts = (hacts > level_at_99).float() # indicator
        return tally.conditional_samples(iacts, seg)
    
    pbar.descnext('condi99')
    condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,
            num_workers=3, pin_memory=True,
            cachefile=tally_cache_file('condi99.npz'))

    # Now summarize the iou stats and graph the units
    iou_99 = tally.iou_from_conditional_indicator_mean(condi99)

    unit_label_99 = [
            (concept.item(), seglabels[concept],
                segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*iou_99.max(0))]

    print("number of neurons:", len(unit_label_99))
    labelcat_list = [labelcat
            for concept, label, labelcat, iou in unit_label_99
            if iou > iou_threshold]
    print("number of activated neurons (threshold=0.04):", len(labelcat_list))
    print("covering {} concepts:".format(len(set(labelcat_list))))

    save_conceptcat_graph(tally_cache_file('concepts_99.svg'), labelcat_list)
    dump_json_file(tally_cache_file('report.json'), dict(
            header=dict(
                name= "%s %s %s" % (str(results_dir), 'imagenet', seg_name),
                image='concepts_99.svg',
                interpretable=len(labelcat_list),
                iou_threshold=iou_threshold,
                units_num=len(unit_label_99),
                uncovered_concepts=len(set(labelcat_list))
            ),
            units=[
                dict(image='image/unit%d.jpg' % u,
                    unit=u, iou=iou, label=label, cat=labelcat[1])
                for u, (concept, label, labelcat, iou)
                in enumerate(unit_label_99)])
            )
    copy_static_file('report.html', tally_cache_file('+report.html'))
    # tally_cache_file.done();

    return topk, rq




def discriminative(
    model: nn.Module,
    dataset: data.Dataset,
    layer: Optional[Layer] = None,
    device: Optional[Device] = None,
    results_dir: Optional[PathLike] = None,
    viz_dir: Optional[PathLike] = None,
    transform_inputs: transforms.TransformToTuple = transforms.first,
    transform_hiddens: transforms.TransformToTensor = transforms.identity,
    **kwargs: Any,
) -> ActivationStats:
    """Compute exemplars for a discriminative model.

    That is, a model for which image goes in, prediction comes out. Its outputs
    will be interpretted as the neuron activations to track.

    Keyword arguments are forwarded to `compute`.

    Args:
        model (nn.Module): The model to dissect.
        dataset (data.Dataset): Dataset of images used to compute the
            top-activating images.
        layer (Optional[Layer], optional): Track unit activations for this
            layer. If not set, NetDissect will only look at the final output
            of the model. Defaults to None.
        device (Optional[Device], optional): Run all computations on this
            device. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            If set and layer is also set, layer name will be appended to path.
            Defaults to same as `run`.
        viz_dir (Optional[PathLike], optional): Directory to write top image
            visualizations to (e.g., individual png images, lightbox, etc.).
            If set and layer is also set, layer name will be appended to path.
            Defaults to same as `run`.
        transform_inputs (transforms.TransformToTuple, optional): Pass batch
            as *args to this function and use output as *args to model.
            Defaults to identity, i.e. entire batch is passed to model.
        transform_hiddens (transforms.TransformToTensor, optional): Pass hidden
            representations, i.e. the activations, to this function and hand
            result to netdissect. Defaults to identity function, i.e. the raw
            data will be tracked by netdissect.

    Returns:
        ActivationStats: The top-k and quantile statistics for every unit.

    """
    model.to(device)

    def resolve(directory: Optional[PathLike]) -> Optional[pathlib.Path]:
        if directory is not None:
            directory = pathlib.Path(directory)
            directory /= str(layer) if layer is not None else 'outputs'
        return directory

    results_dir = resolve(results_dir)
    viz_dir = resolve(viz_dir)

    with nethook.InstrumentedModel(model) as instr:
        if layer is not None:
            layer = str(layer)
            instr.retain_layer(layer, detach=False)

        def compute_topk_and_quantile(*inputs: Any) -> TensorPair:
            inputs = transform_inputs(*transforms.map_location(inputs, device))
            with torch.no_grad():
                outputs = instr(*inputs)
            hiddens = outputs if layer is None else instr.retained_layer(layer)
            hiddens = transform_hiddens(hiddens)
            batch_size, channels, *_ = hiddens.shape
            activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
            pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
            return pooled, activations

        def compute_activations(*inputs: Any) -> torch.Tensor:
            inputs = transform_inputs(*transforms.map_location(inputs, device))
            with torch.no_grad():
                outputs = instr(*inputs)
            hiddens = outputs if layer is None else instr.retained_layer(layer)
            hiddens = transform_hiddens(hiddens)
            return hiddens

        return compute(compute_topk_and_quantile,
                       compute_activations,
                       dataset,
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       **kwargs)


def generative(
    model: nn.Module,
    dataset: data.Dataset,
    layer: Layer,
    device: Optional[Device] = None,
    results_dir: Optional[PathLike] = None,
    viz_dir: Optional[PathLike] = None,
    transform_inputs: transforms.TransformToTuple = transforms.identities,
    transform_hiddens: transforms.TransformToTensor = transforms.identity,
    transform_outputs: transforms.TransformToTensor = transforms.identity,
    **kwargs: Any,
) -> ActivationStats:
    """Compute exemplars for a generative model of images.

    That is, a model for which representation goes in, image comes out.
    Because of the way these models are structured, we need both the generated
    images and the intermediate activation.

    Keyword arguments are forwarded to `compute`.

    Args:
        model (nn.Module): The model to dissect.
        dataset (data.Dataset): Dataset of representations used to generate
            images. The top-activating images will be taken from them.
        layer (Layer): Track unit activations for this layer.
        device (Optional[Device], optional): Run all computations on this
            device. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            If set, layer name will be appended to path. Defaults to same
            as `run`.
        viz_dir (Optional[PathLike], optional): Directory to write top image
            visualizations to (e.g., individual png images, lightbox, etc.).
            If set and layer is also set, layer name will be appended to path.
            Defaults to same as `run`.
        transform_inputs (transforms.TransformToTuple, optional): Pass batch
            as *args to this function and use output as *args to model.
            Defaults to identity, i.e. entire batch is passed to model.
        transform_hiddens (transforms.TransformToTensor, optional): Pass output
            of intermediate layer to this function and hand the result to
            netdissect. This is useful if e.g. your model passes info between
            layers as a dictionary or other non-Tensor data type.
            Defaults to the identity function, i.e. the raw data will be
            tracked by netdissect.
        transform_outputs (transforms.TransformToTensor, optional): Pass output
            of entire model, i.e. generated images, to this function and hand
            result to netdissect. Defaults to identity function, i.e. the raw
            data will be tracked by netdissect.

    Returns:
        ActivationStats: The top-k and quantile statistics for every unit.

    """
    if results_dir is not None:
        results_dir = pathlib.Path(results_dir) / str(layer)
    if viz_dir is not None:
        viz_dir = pathlib.Path(viz_dir) / str(layer)

    model.to(device)
    with nethook.InstrumentedModel(model) as instr:
        instr.retain_layer(layer, detach=False)

        def make_upfn(*inputs: Any):
            x, y = inputs
            print(x.shape, y.shape)
            inputs = (x[None, ...], y[None, ...])
            prob_data = transform_inputs(*transforms.map_location(inputs, device))
            out = model(*prob_data)
            hidden = transform_hiddens(instr.retained_layer(layer))
            print("hidden shape: {}, output img shape: {}".format(hidden.shape, out.shape))
            return upsample.upsampler(
                (64, 64),
                data_shape=hidden.shape[2:],
                image_size=out.shape[2:])

        def compute_topk_and_quantile(*inputs: Any) -> TensorPair:
            inputs = transform_inputs(*transforms.map_location(inputs, device))
            with torch.no_grad():
                model(*inputs)
            hiddens = transform_hiddens(instr.retained_layer(layer))
            batch_size, channels, *_ = hiddens.shape
            activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
            pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
            return pooled, activations

        def compute_activations(*inputs: Any) -> TensorPair:
            inputs = transform_inputs(*transforms.map_location(inputs, device))
            with torch.no_grad():
                images = model(*inputs)
            hiddens = transform_hiddens(instr.retained_layer(layer))
            images = transform_outputs(images)
            return hiddens, images


        return compute(compute_topk_and_quantile,
                       compute_activations,
                       dataset,
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       upfn=make_upfn(*dataset[0]),
                       **kwargs)




def diffusion(
    sampler: Any,
    dataset: data.Dataset,
    layer: Layer,
    ddim_steps: int = 20,
    ddim_eta: float = 0.0,
    scale: float = 3.0,
    save_interval: int = 10,
    device: Optional[Device] = None,
    results_dir: Optional[PathLike] = None,
    viz_dir: Optional[PathLike] = None,
    transform_inputs: transforms.TransformToTuple = transforms.identities,
    transform_hiddens: transforms.TransformToTensor = transforms.identity,
    transform_outputs: transforms.TransformToTensor = transforms.identity,
    **kwargs: Any,
) -> ActivationStats:
    """Compute exemplars for a difussion model of images.

    That is, a model for which representation goes in, image comes out.
    Because of the way these models are structured, we need both the generated
    images and the intermediate activation.

    Keyword arguments are forwarded to `compute`.

    Args:
        sampler (Any): ldm sampler.
        dataset (data.Dataset): Dataset of representations used to generate
            images. The top-activating images will be taken from them.
        layer (Layer): Track unit activations for this layer.
        ddim_steps: number of difussion steps
        ddim_eta: ddim_eta for quality of generated images
        scale: for unconditional guidance
        device (Optional[Device], optional): Run all computations on this
            device. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            If set, layer name will be appended to path. Defaults to same
            as `run`.
        viz_dir (Optional[PathLike], optional): Directory to write top image
            visualizations to (e.g., individual png images, lightbox, etc.).
            If set and layer is also set, layer name will be appended to path.
            Defaults to same as `run`.
        transform_inputs (transforms.TransformToTuple, optional): Pass batch
            as *args to this function and use output as *args to model.
            Defaults to identity, i.e. entire batch is passed to model.
        transform_hiddens (transforms.TransformToTensor, optional): Pass output
            of intermediate layer to this function and hand the result to
            netdissect. This is useful if e.g. your model passes info between
            layers as a dictionary or other non-Tensor data type.
            Defaults to the identity function, i.e. the raw data will be
            tracked by netdissect.
        transform_outputs (transforms.TransformToTensor, optional): Pass output
            of entire model, i.e. generated images, to this function and hand
            result to netdissect. Defaults to identity function, i.e. the raw
            data will be tracked by netdissect.

    Returns:
        ActivationStats: The top-k and quantile statistics for every unit.

    """
    if results_dir is not None:
        results_dir = pathlib.Path(results_dir) / str(layer)
    if viz_dir is not None:
        viz_dir = pathlib.Path(viz_dir) / str(layer)

    model = sampler.model
    model.to(device)

    num_classes = model.num_classes
    cal_e_with_uncond = lambda e_t, e_t_uncond: e_t_uncond + scale * (e_t - e_t_uncond)
    with nethook.InstrumentedModel(model, retain_list=True, save_interval=save_interval, max_step=ddim_steps) as instr:
        instr.retain_layer(layer, detach=False)

        def diffusion(*inputs: Any):
            x_T, xc = inputs
            img_num = x_T.shape[0]
            # print(x_T.shape, xc.shape)
            with model.ema_scope():
                uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(img_num * [num_classes]).cuda()})
                c = model.get_learned_conditioning({model.cond_stage_key: xc.cuda()})

                samples_ddim, intermediates = sampler.sample(S=ddim_steps, x_T=x_T.cuda(), conditioning=c,
                                                             batch_size=img_num, shape=[3, 64, 64],
                                                             verbose=False, log_every_t=save_interval,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc,
                                                             eta=ddim_eta)
            return samples_ddim, intermediates

        def decode_latents(samples_ddim):
            samples_ddim = model.decode_first_stage(samples_ddim)
            # samples_ddim = torch.clamp((samples_ddim + 1.0) / 2.0,
            #                            min=0.0, max=1.0)
            return samples_ddim

        def compute_topk_and_quantile(*inputs: Any) -> TensorPair:
            # here we stack the output of images, so there would be [saved_steps * batch_size, channels, w, h]
            with torch.no_grad():
                diffusion(*inputs)
            hiddens = instr.retained_layer(layer, clear=True)  # only remaining conditional e_t
            # hiddens = torch.cat([cal_e_with_uncond(*h_t.chunk(2)) for h_t in hiddens])
            hiddens = [cal_e_with_uncond(*h_t.chunk(2)) for h_t in hiddens][-1]

            batch_size, channels, *_ = hiddens.shape
            activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
            pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
            return pooled, activations

        def compute_activations(*inputs: Any) -> TensorPair:
            # here we stack the output of images, so there would be [saved_steps * batch_size, channels, w, h]
            with torch.no_grad():
                images, intermediates = diffusion(*inputs)
            # predict_imgs = torch.cat(
                # intermediates['pred_x0'][1:])  # direct denoised output based on specific neuron out
            predict_imgs = intermediates['pred_x0'][-1]  # direct denoised output based on specific neuron out
            predict_imgs = decode_latents(predict_imgs)

            hiddens = instr.retained_layer(layer, clear=True)  # only remaining conditional e_t
            # hiddens = torch.cat([cal_e_with_uncond(*h_t.chunk(2)) for h_t in hiddens])
            hiddens = [cal_e_with_uncond(*h_t.chunk(2)) for h_t in hiddens][-1]

            return hiddens, predict_imgs

        def make_upfn(*inputs: Any):
            x, y = inputs
            print(x.shape, y.shape)
            inputs = (x[None, ...], y[None, ...])
            hidden, out = compute_activations(*inputs)
            print("hidden shape: {}, output img shape: {}".format(hidden.shape, out.shape))
            return upsample.upsampler(
                (64, 64),
                data_shape=hidden.shape[2:],
                image_size=out.shape[2:])

        # if 'batch_size' in kwargs:
        #     kwargs['batch_size'] = kwargs['batch_size'] * 3

        return compute(compute_topk_and_quantile,
                       compute_activations,
                       dataset,
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       upfn=make_upfn(*dataset[0]),
                       **kwargs)








import json
from collections import defaultdict
def graph_conceptcatlist(conceptcatlist, **kwargs):
    count = defaultdict(int)
    catcount = defaultdict(int)
    for c in conceptcatlist:
        count[c] += 1
    for c in count.keys():
        catcount[c[1]] += 1
    cats = ['object', 'part', 'material', 'texture', 'color']
    catorder = dict((c, i) for i, c in enumerate(cats))
    sorted_labels = sorted(count.keys(),
        key=lambda x: (catorder[x[1]], -count[x]))
    sorted_labels
    return bargraph.make_svg_bargraph(
        [label for label, cat in sorted_labels],
        [count[k] for k in sorted_labels],
        [(c, catcount[c]) for c in cats], **kwargs)

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)

def dump_json_file(target, data):
    with open(target, 'w') as f:
        json.dump(data, f, indent=1, cls=FloatEncoder)

def copy_static_file(source, target):
    sourcefile = os.path.join(
            os.path.dirname(inspect.getfile(netdissect)), source)
    shutil.copy(sourcefile, target)





class FloatEncoder(json.JSONEncoder):
    def __init__(self, nan_str='"NaN"', **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        def floatstr(o, allow_nan=self.allow_nan,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY,
                nan_str=self.nan_str):
            if o != o:
                text = nan_str
            elif o == _inf:
                text = '"Infinity"'
            elif o == _neginf:
                text = '"-Infinity"'
            else:
                return repr(o)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))
            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)