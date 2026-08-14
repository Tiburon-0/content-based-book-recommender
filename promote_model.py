'''Model Registry Promotion'''

import argparse

import wandb

ENTITY = 'tiburon_0-university-of-denver'
PROJECT = 'content-based-book-recommender'
ARTIFACT_NAME = 'tiburon-book-recommender'
ARTIFACT_TYPE = 'model'

REGISTRY_PATH = 'wandb-registry-model/tiburon-book-recommender'

PROMOTION_ALIASES = ('staging', 'production')

def qualified_name(version):
    '''Builds the fully-qualified artifact path W&B expects: entity/project/name:version'''

    return f'{ENTITY}/{PROJECT}/{ARTIFACT_NAME}:{version}'

def list_versions():
    '''Prints every logged version of the model artifact, newest first'''

    api = wandb.Api()

    versions = api.artifacts(
        type_name=ARTIFACT_TYPE,
        name=f'{ENTITY}/{PROJECT}/{ARTIFACT_NAME}',
        order='-created_at',
    )

    version_list = list(versions)

    print(f'\n{len(version_list)} logged version(s) of {ARTIFACT_NAME}:\n')

    for artifact in version_list:
        alias_list = ', '.join(artifact.aliases) if artifact.aliases else '-'

        catalog = artifact.metadata.get('catalog_size', '?')
        vocabulary = artifact.metadata.get('vocabulary_size', '?')
        min_df = artifact.metadata.get('min_df', '?')

        print(f'{artifact.version:>5} | aliases: {alias_list:<28} | '
              f'catalog: {catalog:>8} | vocabulary: {vocabulary:>10} | min_df: {min_df}')

    print()

    return version_list

def promote(version, alias, registry_path=REGISTRY_PATH):
    '''Links one logged artifact version into the registry'''

    if alias not in PROMOTION_ALIASES:
        raise ValueError(f'{alias!r} is not a recognized stage; expected either {PROMOTION_ALIASES[0]} or {PROMOTION_ALIASES[1]}')

    api = wandb.Api()

    print(f'Fetching {qualified_name(version)}...')

    artifact = api.artifact(qualified_name(version), type=ARTIFACT_TYPE)

    print(f'Found {artifact.name} | size: {artifact.size / 1e6:,.1f} MB | metadata: {artifact.metadata}')

    print(f'Linking to {registry_path} as {alias!r}...')

    artifact.link(target_path=registry_path, aliases=[alias])

    print(f'Promoted {artifact.name} -> {registry_path}:{alias}')

    return artifact

def main():
    '''Parses command-line flags and dispatches to listing or promotion'''

    parser = argparse.ArgumentParser(
        description='Promote a logged model artifact into the W&B model registry'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='list every logged version with its aliases and metadata, then exit'
    )

    parser.add_argument(
        '--version',
        default='latest',
        help="artfiact version to promote, e.g. 'v3' (default: latest)"
    )

    parser.add_argument(
        '--alias',
        default='staging',
        choices=PROMOTION_ALIASES,
        help='registry stage to assign (default: staging)'
    )

    args = parser.parse_args()

    if args.list:
        list_versions()
        return

    promote(args.version, args.alias)

if __name__ == '__main__':
    main()
