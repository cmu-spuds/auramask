import enum
import keras
import os

IPFS_GATEWAY = "https://ipfs.meekoracc.com/ipfs/"


class StyleRefs(enum.Enum):
    STARRYNIGHT = {
        "source": "url",
        "url": "https://keras.io/img/examples/generative/neural_style_transfer/neural_style_transfer_5_1.jpg",
    }
    HOPE = {
        "source": "url",
        "url": "https://i.pinimg.com/236x/e7/b3/46/e7b346bc9b0dae896705516bb7258cb6--obama-poster-design-tutorials.jpg?nii=t",
    }
    DIM = {
        "source": "url",
        "url": "https://images.unsplash.com/photo-1719066373323-c3712474b2a4",
    }
    ABSTRACT = {
        "source": "ipfs",
        "url": "UNKNOWN",
        "cid": "Qmdg9UbTTQS7nXqLJv3hKhDRVinCQXR3H6asU462MATCzd",
    }
    VICTONGAI_0 = {
        "source": "ipfs",
        "url": "https://uploads1.wikiart.org/00231/images/victo-ngai/5918-193068.jpg",
        "cid": "QmXkzooMZWV6Q4rR6HJmFovzsGEK9hz9WaWt4sxiwCHN2X",
    }
    VICTONGAI_1 = {
        "source": "ipfs",
        "url": "https://uploads1.wikiart.org/00231/images/victo-ngai/5918-203486.jpg",
        "cid": "Qmd6Yc9pYf2r4nBZ4zos7JPEwxvXGNnumEb9QkQhDfkWuC",
    }
    JANETFISH_0 = {
        "source": "ipfs",
        "url": "https://uploads3.wikiart.org/images/janet-fish/self-portrait.jpg",
        "cid": "QmcmzSDXrwecUkPzxV4jecwBQhy2DcNZRAArzi9FY7WZMd",
    }
    ATSUKOTANAKA_0 = {
        "source": "ipfs",
        "url": "https://uploads6.wikiart.org/images/atsuko-tanaka/untitled-1999.jpg",
        "cid": "QmXmsAx5xqPbu4LwB8h7jUoPnRaaqx56bFYYNNm7YDakTU",
    }
    KAZUOSHIRAGA_0 = {
        "source": "ipfs",
        "url": "https://uploads6.wikiart.org/images/kazuo-shiraga/funsyutu-1997.jpg",
        "cid": "Qmakp7b8xndH2n7PwMoeND4qxF51Ve8W26UReN9pLHcuDm",
    }
    CHARLESGIBBONS_0 = {
        "source": "ipfs",
        "url": "https://uploads5.wikiart.org/00410/images/charles-gibbons/after-the-rain-4-2010.jpg",
        "cid": "QmVAkiNJ4jnY97CaJbVzEJzcfCrkL4FLr1sLet4cAeW5RY",
    }
    CHARLESGIBBONS_1 = {
        "source": "ipfs",
        "url": "https://uploads5.wikiart.org/00217/images/charles-gibbons/gibbons-2007-vernazza.jpg",
        "cid": "QmZ1BRxUx4xRFkEkJZUVEH6XCfSxxwrbHPghUNubn8JKqi",
    }
    BORISKUSTODIEV_0 = {
        "source": "ipfs",
        "url": "https://www.wikiart.org/en/boris-kustodiev/meeting-easter-day-1917",
        "cid": "QmP9Hij1GZNTHfmEaHVwk3mYdfe5RQYWdmKyxeKSmbbj7z",
    }
    WARHOL_0 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmbDpzEHJexz99ko6fJY9YgysniKBsfjPLRedc5Yj3uGxT?filename=warhol.jpg",
        "cid": "QmbDpzEHJexz99ko6fJY9YgysniKBsfjPLRedc5Yj3uGxT",
    }
    CHE_0 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmUgSxHqtT54c6qnnFHRJvahT9yGP2s86fiW7hCFPnWtL8?filename=che-1.jpg",
        "cid": "QmUgSxHqtT54c6qnnFHRJvahT9yGP2s86fiW7hCFPnWtL8",
    }
    CHE_1 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmPz4rwRCrVLEwmKgtFDiCMWiuyMLBmPYn37kDUGNzk3o6?filename=che-2.jpg",
        "cid": "QmPz4rwRCrVLEwmKgtFDiCMWiuyMLBmPYn37kDUGNzk3o6",
    }
    CHE_2 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmSLH67ipyESm2d2717ea23983xcV5m7fP3XXRiFZdN7Pa?filename=che-3.jpg",
        "cid": "QmSLH67ipyESm2d2717ea23983xcV5m7fP3XXRiFZdN7Pa",
    }
    HARING_0 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmannyGdwCEN8cD51FJNA5V6DGbeH6dPSwyVh8bfSp7Coc?filename=keith-haring.jpg",
        "cid": "QmannyGdwCEN8cD51FJNA5V6DGbeH6dPSwyVh8bfSp7Coc",
    }
    HARING_1 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmQWhEqoRNzazemE3xH3Cq5Xx9uQbzZmMHZBcJnNZx6HSD?filename=keith-dog.jpg",
        "cid": "QmQWhEqoRNzazemE3xH3Cq5Xx9uQbzZmMHZBcJnNZx6HSD",
    }
    HARING_2 = {
        "source": "ipfs",
        "url": "https://ipfs.io/ipfs/QmXkEcjg4xiSq9LtimBsCVxRtixy88kJWxQ7f2pXa4UPAZ?filename=keith-man.jpg",
        "cid": "QmXkEcjg4xiSq9LtimBsCVxRtixy88kJWxQ7f2pXa4UPAZ",
    }
    HARING_3 = {
        "source": "ipfs",
        "url": "https://media.kjzz.org/s3fs-public/styles/juicebox_large/public/keith-haring-painting-full-20200624.jpg?itok=HCFAAiJd",
        "cid": "QmaXEmiGHKrJ6GZSUvUhjJjCWXTA3ru93G58goREiovucF",
    }
    HARING_4 = {
        "source": "ipfs",
        "url": "https://www.itl.cat/pngfile/big/128-1285387_keith-haring-2-keith-haring-wallpaper6-madonna-into.jpg",
        "cid": "Qma3i63Ngg8gWkxa1BhtmtJJvWHZBU5zWg9LgTiHdZLX9p",
    }

    def get_img(self):
        if self.value["source"] == "url":
            style_reference_image_path = keras.utils.get_file(
                "%s.jpg" % self.name, self.value["url"], cache_subdir="style"
            )
        elif self.value["source"] == "ipfs":
            style_reference_image_path = keras.utils.get_file(
                "%s.jpg" % self.name,
                os.path.join(IPFS_GATEWAY, self.value["cid"]),
                cache_subdir="style",
            )

        img = keras.utils.load_img(
            style_reference_image_path, target_size=(256, 256), keep_aspect_ratio=True
        )
        # Style reference
        img = keras.utils.img_to_array(img)
        img = keras.ops.expand_dims(img, axis=0)
        return img
