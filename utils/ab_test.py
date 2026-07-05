import hashlib


class ABTestManager:
    """Manages deterministic user cohort allocation and algorithm assignment."""

    @staticmethod
    def get_cohort(user_id: str) -> str:
        """Deterministically map a user to cohort 'A' or cohort 'B'.

        Cohort 'A' (Control) receives blended recommendations.
        Cohort 'B' (Treatment) receives sequential-heavy recommendations.
        """
        if not user_id:
            return "A"
        # Deterministic hashing
        hash_val = int(hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16)
        return "B" if hash_val % 2 == 1 else "A"
